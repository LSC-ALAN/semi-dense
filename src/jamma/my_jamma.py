import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange
from src.jamma.utils.utils import KeypointEncoder_wo_score, up_conv4, MLPMixerEncoderLayer, normalize_keypoints
from src.jamma.mamba_module import JointMamba
from src.jamma.matching_module import CoarseMatching
from src.utils.profiler import PassThroughProfiler
from kornia.utils import create_meshgrid

torch.backends.cudnn.deterministic = True
INF = 1E9


class FineSubMatching1(nn.Module):
    """Fine-level and Sub-pixel matching"""

    def __init__(self, config, profiler):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.inference = config['fine']['inference']
        dim_f = 64
        self.fine_thr = config['fine']['thr']
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(2 * dim_f, 4, bias=False))
        self.fine_spv_max = None  # saving memory
        self.profiler = profiler

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        M, WW, C = feat_f0_unfold.shape
        W_f = self.W_f

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
                'mkpts0_f_train': data['mkpts0_c_train'],
                'mkpts1_f_train': data['mkpts1_c_train'],
                'conf_matrix_fine': torch.zeros(1, W_f * W_f, W_f * W_f, device=feat_f0_unfold.device),
                'b_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                'i_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                'j_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
            })
            return

        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)

        # normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_f0, feat_f1])
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
                                  feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(**self.get_fine_sub_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data))

    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data):
        with torch.no_grad():
            W_f = self.W_f

            # 1. confidence thresholding
            mask = conf_matrix_fine > self.fine_thr

            if mask.sum() == 0:
                mask[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1

            # match only the highest confidence
            mask = mask \
                   * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

            # 3. find all valid fine matches
            # this only works when at most one `True` in each row
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            # 4. update with matches in original image resolution

            # indices from coarse matches
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # scale (coarse level / fine-level)
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # coarse level matches scaled to fine-level (1/2) xy
            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # updated b_ids after second thresholding
            updated_b_ids = b_ids_c[b_ids]

            # scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # fine-level discrete matches on window coordiantes
            mkpts0_f_window = torch.stack(
                [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
                dim=1)

            mkpts1_f_window = torch.stack(
                [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
                dim=1)

        # sub-pixel refinement
        sub_ref = self.subpixel_mlp(torch.cat([feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]], dim=-1))
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
        sub_ref0, sub_ref1 = sub_ref0.squeeze(1), sub_ref1.squeeze(1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        pad = 0 if W_f % 2 == 0 else W_f // 2
        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f1 = (mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad) * scale0  # + sub_ref0
        mkpts1_f1 = (mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad) * scale1  # + sub_ref1
        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale0  # + sub_ref0
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale1  # + sub_ref1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches is the current prediction (for visualization)
        sub_pixel_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],  # mconf == 0 => gt matches
            'mkpts0_f1': mkpts0_f1[mconf != 0],
            'mkpts1_f1': mkpts1_f1[mconf != 0],
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0]
        }

        # These matches are used for training
        if not self.inference:
            sub_pixel_matches.update({
                'mkpts0_f_train': mkpts0_f_train,
                'mkpts1_f_train': mkpts1_f_train,
                'b_ids_fine': data['b_ids'],
                'i_ids_fine': data['i_ids'],
                'j_ids_fine': data['j_ids'],
                'conf_matrix_fine': conf_matrix_fine
            })

        return sub_pixel_matches


class FineSubMatching(nn.Module):
    def __init__(self, config, profiler,in_dim=64, hidden_dim=128, num_heads=4, depth=2):
        super().__init__()
        self.W_f = config['fine_window_size']
        self.inference = config['fine']['inference']
        self.fine_thr = config['fine']['thr']
        self.fine_spv_max = None  # saving memory
        self.profiler = profiler
        self.input_proj = nn.Conv2d(in_dim * 2, hidden_dim, kernel_size=1)
        dim_f = 64

        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)

        self.pos_embed = nn.Parameter(torch.randn(1, 25, hidden_dim))  # learnable pos

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_flow_certainty = nn.Linear(hidden_dim, 3)  # dx, dy, certainty

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        M, WW, C = feat_f0_unfold.shape  # WW=25
        W_f = self.W_f  # e.g., 5
        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
                'mkpts0_f_train': data['mkpts0_c_train'],
                'mkpts1_f_train': data['mkpts1_c_train'],
                'conf_matrix_fine': torch.zeros(1, W_f * W_f, W_f * W_f, device=feat_f0_unfold.device),
                'b_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                'i_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                'j_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
            })
            return
        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)
        # normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_f0, feat_f1])

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(**self.get_fine_sub_match(feat_f0, feat_f1, data))


    # def get_label(self,w,h,b_ids_c, i_ids_c, j_ids_c):



    def get_fine_sub_match(self,feat_f0_unfold, feat_f1_unfold, data):
        W_f = self.W_f  # e.g., 5
        device = data['imagec_0'].device
        N, _, H0, W0 = data['imagec_0'].shape
        _, _, H1, W1 = data['imagec_1'].shape
        scale = 2
        scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
        h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
        scale_f_c = 4
        # 2. get coarse prediction
        b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
        # 2. warp grids
        # create kpts in meshgrid and resize them to image resolution
        grid_pt0_c = create_meshgrid(h0, w0, False, device).repeat(N, 1, 1, 1)
        grid_pt0_i = scale0[:, None, ...] * grid_pt0_c
        grid_pt1_c = create_meshgrid(h1, w1, False, device).repeat(N, 1, 1, 1)
        grid_pt1_i = scale1[:, None, ...] * grid_pt1_c

        # unfold (crop windows) all local windows
        stride_f = data['hw0_f'][0] // data['hw0_c'][0]

        pad = 0 if W_f % 2 == 0 else W_f // 2
        grid_pt0_i = rearrange(grid_pt0_i, 'n h w c -> n c h w')
        grid_pt0_i = F.unfold(grid_pt0_i, kernel_size=(W_f, W_f), stride=stride_f, padding=pad)
        grid_pt0_i = rearrange(grid_pt0_i, 'n (c ww) l -> n l ww c', ww=W_f ** 2)
        grid_pt0_i = grid_pt0_i[b_ids, i_ids]

        grid_pt1_i = rearrange(grid_pt1_i, 'n h w c -> n c h w')
        grid_pt1_i = F.unfold(grid_pt1_i, kernel_size=(W_f, W_f), stride=stride_f, padding=pad)
        grid_pt1_i = rearrange(grid_pt1_i, 'n (c ww) l -> n l ww c', ww=W_f ** 2)
        grid_pt1_i = grid_pt1_i[b_ids, j_ids]
        data.update({
            "grid_pt0_i": grid_pt0_i,
            "grid_pt1_i": grid_pt1_i
        })
        M, WW, C = feat_f0_unfold.shape  # WW=25

        # 1. 拼接两个 patch（A 和 B）
        feat_cat_AB = torch.cat([feat_f0_unfold, feat_f1_unfold], dim=-1)  # (M, 25, 2C)

        # 2. 输入 projection
        x_AB = self.input_proj(feat_cat_AB.permute(0, 2, 1).reshape(M, 2*C, W_f, W_f))  # (M, hidden_dim, 5, 5)
        x_AB = x_AB.flatten(2).permute(0, 2, 1)  # (M, 25, hidden_dim)

        # 3. 添加可学习的位置编码
        x_AB = x_AB + self.pos_embed  # (M, 25, hidden_dim)

        # 4. 输入 Transformer
        x_AB = self.transformer(x_AB)  # (M, 25, hidden_dim)

        # 5. 输出三通道：dx, dy, certainty
        out_AB = self.to_flow_certainty(x_AB)  # (M, 25, 3)

        # 6. 分离 flow 与 certainty
        flow0_1 = 2.5 * torch.tanh(out_AB[:, :, :2])       # shape: (M, 25, 2)
        certainty0_1 = torch.sigmoid(out_AB[:, :, 2:])  # shape: (M, 25, 1)
        flow0_1_flat = flow0_1.reshape(-1, 2)  # (M*25, 2)
        certainty0_1_flat = certainty0_1.reshape(-1)  # (M*25,)

        feat_cat_BA = torch.cat([feat_f1_unfold, feat_f0_unfold], dim=-1)  # (M, 25, 2C)

        # 2. 输入 projection
        x_BA = self.input_proj(feat_cat_BA.permute(0, 2, 1).reshape(M, 2 * C, W_f, W_f))  # (M, hidden_dim, 5, 5)
        x_BA = x_BA.flatten(2).permute(0, 2, 1)  # (M, 25, hidden_dim)

        # 3. 添加可学习的位置编码
        x_BA = x_BA + self.pos_embed  # (M, 25, hidden_dim)

        # 4. 输入 Transformer
        x_BA = self.transformer(x_BA)  # (M, 25, hidden_dim)

        # 5. 输出三通道：dx, dy, certainty
        out_BA = self.to_flow_certainty(x_BA)  # (M, 25, 3)

        # 6. 分离 flow 与 certainty
        flow1_0 = 2.5 * torch.tanh(out_BA[:, :, :2])  # shape: (M, 25, 2)
        certainty1_0 = torch.sigmoid(out_BA[:, :, 2:])  # shape: (M, 25, 1)
        flow1_0_flat = flow1_0.reshape(-1, 2)  # (M*25, 2)
        certainty1_0_flat = certainty1_0.reshape(-1)  # (M*25,)

        with torch.no_grad():
            # indices from coarse matches
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # scale (coarse level / fine-level)
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # coarse level matches scaled to fine-level (1/2) xy
            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # updated b_ids after second thresholding
            updated_b_ids = b_ids_c

            # scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

        # 2. 扩展 coarse 坐标，broadcast + reshape
        mkpts1_c_flat = mkpts1_c_scaled_to_f[:, None, :].expand(-1, 25, -1).reshape(-1, 2)  # (M*25, 2)

        # get train data
        mkpts1_f_train = mkpts1_c_flat + flow0_1_flat  # (M*25, 2)

        # 3. 筛选 high-confidence   mask = bids*25
        thr = 0.5
        mask1 = certainty0_1_flat > thr  # (M*25,)
        flow0_1_valid = flow0_1_flat[mask1]  # (?, 2)
        mkpts1_c_valid = mkpts1_c_flat[mask1]  # (?, 2)

        # 4. 得到最终 fine-level 点
        mkpts1_f_valid = mkpts1_c_valid + flow0_1_valid  # (?, 2)
        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement) M 2
        scale1_expand = scale1[:, None, :].expand(-1, 25, -1).reshape(-1, 2)  # (M*25, 2)
        scale1_valid = scale1_expand[mask1]  # (?, 2)
        mkpts1_f = mkpts1_f_valid * scale1_valid  # + sub_ref0
        mkpts1_f = mkpts1_f.clone().detach()
        cell0 = data['grid_pt0_i'].reshape(-1, 2)[mask1]






        # 2. 扩展 coarse 坐标，broadcast + reshape
        mkpts0_c_flat = mkpts0_c_scaled_to_f[:, None, :].expand(-1, 25, -1).reshape(-1, 2)  # (M*25, 2)

        # get train data
        mkpts0_f_train = mkpts0_c_flat + flow1_0_flat  # (M*25, 2)

        # 3. 筛选 high-confidence   mask = bids*25
        thr = 0.5
        mask0 = certainty1_0_flat > thr  # (M*25,)
        flow0_1_valid = flow1_0_flat[mask0]  # (?, 2)
        mkpts0_c_valid = mkpts0_c_flat[mask0]  # (?, 2)

        # 4. 得到最终 fine-level 点
        mkpts0_f_valid = mkpts0_c_valid + flow0_1_valid  # (?, 2)
        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement) M 2
        scale0_expand = scale0[:, None, :].expand(-1, 25, -1).reshape(-1, 2)  # (M*25, 2)
        scale0_valid = scale0_expand[mask0]  # (?, 2)
        mkpts0_f = mkpts0_f_valid * scale0_valid  # + sub_ref0
        mkpts0_f = mkpts0_f.clone().detach()
        cell1 = data['grid_pt1_i'].reshape(-1, 2)[mask0]

        # These matches is the current prediction (for visualization)
        sub_pixel_matches = {
            'mkpts0_f': cell0,
            'mkpts1_f': mkpts1_f,
            'mask_predict1': mask1
        }
        sub_pixel_matches.update({
            'mkpts0_f_train': mkpts0_f_train,
            'mkpts1_f_train': mkpts1_f_train,
            'mkpts0_f_inference': mkpts0_f_valid,
            'mkpts1_f_inference': mkpts1_f_valid,
            'cell0': cell0,
            'cell1': cell1,

            'certainty0_1_flat': certainty0_1_flat,  # (M*25,1)
            'certainty1_0_flat': certainty1_0_flat,  # (M*25,1)
            'mask_predict0': mask0,  # (M*25,1)
            'b_ids_fine': data['b_ids'],
            'i_ids_fine': data['i_ids'],
            'j_ids_fine': data['j_ids'],
        })
        # # These matches are used for training
        # if not self.inference:
        #     sub_pixel_matches.update({
        #         'mkpts0_f_train': mkpts0_f_train,
        #         'mkpts1_f_train': mkpts1_f_train,
        #         'mkpts0_f_train_2': mkpts0_f_valid,
        #         'mkpts1_f_train_2': mkpts1_f_valid,
        #         'cell0': cell0,
        #         'cell1': cell1,
        #
        #         'certainty0_1_flat': certainty0_1_flat, # (M*25,1)
        #         'certainty1_0_flat': certainty1_0_flat,  # (M*25,1)
        #         'mask_predict0': mask0,  # (M*25,1)
        #         'b_ids_fine': data['b_ids'],
        #         'i_ids_fine': data['i_ids'],
        #         'j_ids_fine': data['j_ids'],
        #     })

        return sub_pixel_matches

class JamMa(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = self.config['coarse']['d_model']
        self.d_model_f = self.config['fine']['d_model']

        self.kenc = KeypointEncoder_wo_score(self.d_model_c, [32, 64, 128, self.d_model_c])
        self.joint_mamba = JointMamba(self.d_model_c, 4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, profiler=self.profiler)
        self.coarse_matching = CoarseMatching(config['match_coarse'], self.profiler)

        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2*dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        W = self.config['fine_window_size']
        self.fine_enc = nn.ModuleList([MLPMixerEncoderLayer(2*W**2, 64) for _ in range(4)])
        self.fine_matching = FineSubMatching(config, self.profiler)

    def coarse_match(self, data):
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['imagec_0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['imagec_1'].shape[-2:])

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        with self.profiler.profile("coarse interaction"):
            self.joint_mamba(data)

        # 3. match coarse-level
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        with self.profiler.profile("coarse matching"):
            self.coarse_matching(data['feat_8_0'].transpose(1,2), data['feat_8_1'].transpose(1,2), data, mask_c0=mask_c0, mask_c1=mask_c1)

    def inter_fpn(self, feat_8, feat_4):
        d2 = self.up2(feat_8)  # 1/4
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
        feat_4 = self.act(self.conv7b(d2))

        d1 = self.up3(feat_4)  # 1/2
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        return feat_2

    def fine_preprocess(self, data, profiler):
        data['resolution1'] = 8
        stride = data['resolution1'] // self.config['resolution'][1]
        W = self.config['fine_window_size']
        feat_8 = torch.cat([data['feat_8_0'], data['feat_8_1']], 0).view(2*data['bs'], data['c'], data['h_8'], -1)
        feat_4 = torch.cat([data['feat_4_0'], data['feat_4_1']], 0)

        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            feat1 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            return feat0, feat1

        # feat_f = self.inter_fpn(feat_8, feat_4, feat_2)
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        data.update({'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

        # 1. unfold(crop) all local windows
        pad = 0 if W % 2 == 0 else W//2
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)  # [b, h_f/stride * w_f/stride, w*w, c]

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]  # [n, ww, cf]

        feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], 1).transpose(1, 2)
        for layer in self.fine_enc:
            feat_f = layer(feat_f)
        feat_f0_unfold, feat_f1_unfold = feat_f[:, :, :W**2], feat_f[:, :, W**2:]
        return feat_f0_unfold, feat_f1_unfold

    def forward(self, data, mode='test'):
        self.mode = mode
        """Run SuperGlue on a pair of keypoints and descriptors"""
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })

        self.coarse_match(data)

        with self.profiler.profile("fine matching"):
            # 4. fine-level matching module
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data, self.profiler)

            # 5. match fine-level and sub-pixel refinement
            self.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)
