import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_max


class Trans4Map(nn.Module):
    def __init__(self, cfg, device):
        super(Trans4Map, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']
        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = torch.device('cpu')  # cpu

        if mem_update == 'lstm':
            self.rnn = nn.LSTMCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)

        elif mem_update == 'gru':
            self.rnn = nn.GRUCell(ego_feat_dim, mem_feat_dim, bias=True)
            self.rnn_r = nn.GRUCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)
            self.rnn_r.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn_r.weight_hh)
            self.rnn_r.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn_r.weight_ih)
            self.rnn_r.bias_hh.data = torch.zeros_like(self.rnn_r.bias_hh)  # redundant with bias_ih
            self.rnn_r.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn_r.bias_ih)

        elif mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)
        else:
            raise Exception('{} memory update not supported.'.format(mem_update))

        self.fuse = nn.Conv2d(mem_feat_dim * 2, mem_feat_dim, 1, 1, 0)
        self.decoder = Decoder(mem_feat_dim, n_obj_classes)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def encode(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        T, C, H, W = features.shape

        mask_inliers = ~mask_outliers

        memory_size = map_height * map_width * self.mem_feat_dim * 4 / 1e9
        if memory_size > 5:
            self.device_mem = torch.device('cpu')
        else:
            self.device_mem = torch.device('cuda')
        self.decoder = self.decoder.to(self.device_mem)

        if self.mem_update == 'lstm':
            state = (
            torch.zeros((map_height * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
            torch.zeros((map_height * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((map_height * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
            state_r = torch.zeros((map_height * map_width, self.mem_feat_dim), dtype=torch.float,
                                  device=self.device_mem)
        elif self.mem_update == 'replace':
            state = torch.zeros((map_height * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((map_height * map_width), dtype=torch.bool, device=self.device)
        height_map = torch.zeros((map_height * map_width), dtype=torch.float, device=self.device)
        height_map_r = torch.zeros((map_height * map_width), dtype=torch.float, device=self.device)

        for t in tqdm(range(T)):

            feature = features[t, :, :, :]
            feature_r = features[T - t - 1, :, :, :]
            world_to_map = proj_wtm[t, :, :, :]
            world_to_map_r = proj_wtm[T - t - 1, :, :, :]
            inliers = mask_inliers[t, :, :]
            inliers_r = mask_inliers[T - t - 1, :, :]
            height = heights[t, :, :]
            height_r = heights[T - t - 1, :, :]

            world_to_map = world_to_map.long()
            world_to_map_r = world_to_map_r.long()

            feature = feature.to(self.device)
            world_to_map = world_to_map.to(self.device)
            inliers = inliers.to(self.device)
            height = height.to(self.device)

            feature_r = feature_r.to(self.device)
            world_to_map_r = world_to_map_r.to(self.device)
            inliers_r = inliers_r.to(self.device)
            height_r = height_r.to(self.device)

            if self.ego_downsample:
                world_to_map = world_to_map[::4, ::4, :]
                inliers = inliers[::4, ::4]
                height = height[::4, ::4]

            flat_indices = (map_width * world_to_map[:, :, 1] + world_to_map[:, :, 0]).long()
            flat_indices = flat_indices[inliers]
            height = height[inliers]
            height += 1000
            height_map, highest_height_indices = scatter_max(
                height,
                flat_indices,
                dim=0,
                out=height_map,
            )

            flat_indices_r = (map_width * world_to_map_r[:, :, 1] + world_to_map_r[:, :, 0]).long()
            flat_indices_r = flat_indices_r[inliers_r]
            height_r = height_r[inliers_r]
            height_r += 1000
            height_map_r, highest_height_indices_r = scatter_max(
                height_r,
                flat_indices_r,
                dim=0,
                out=height_map_r,
            )

            m = highest_height_indices >= 0
            m_r = highest_height_indices_r >= 0

            observed_masks += m

            if m.any():
                feature = F.interpolate(feature.unsqueeze(0), size=(480, 640), mode="bilinear", align_corners=True)
                feature = feature.squeeze(0)
                if self.ego_downsample:
                    feature = feature[:, ::4, ::4]

                feature = feature.permute(1, 2, 0)  # -- (N,H,W,512)

                feature = feature[inliers, :]

                tmp_memory = feature[highest_height_indices[m], :]

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][m, :].to(self.device),
                                 state[1][m, :].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][m, :] = tmp_state[0].to(self.device_mem)
                    state[1][m, :] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[m, :].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[m, :] = tmp_state.to(self.device_mem)

                elif self.mem_update == 'replace':
                    tmp_memory = self.linlayer(tmp_memory)
                    state[m, :] = tmp_memory.to(self.device_mem)

                else:
                    raise NotImplementedError

                del tmp_memory
            if m_r.any():
                feature_r = F.interpolate(feature_r.unsqueeze(0), size=(480, 640), mode="bilinear", align_corners=True)
                feature_r = feature_r.squeeze(0)

                feature_r = feature_r.permute(1, 2, 0)  # -- (N,H,W,512)

                feature_r = feature_r[inliers_r, :]

                tmp_memory_r = feature_r[highest_height_indices_r[m_r], :]

                tmp_state_r = state_r[m_r, :].to(self.device)

                tmp_state_r = self.rnn_r(tmp_memory_r, tmp_state_r)

                state_r[m_r, :] = tmp_state_r.to(self.device_mem)
                del tmp_memory_r

            del feature, feature_r

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = torch.cat((state, state_r), dim=-1)
        elif self.mem_update == 'replace':
            memory = state

        memory = memory.view(map_height, map_width, self.mem_feat_dim * 2)

        memory = memory.permute(2, 0, 1)
        memory = memory.unsqueeze(0)
        memory = self.fuse(memory)

        return memory, observed_masks, height_map

    def forward(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        # features = self.encoder(rgb)

        memory, observed_masks, height_map = self.encode(features,
                                                         proj_wtm,
                                                         mask_outliers,
                                                         heights,
                                                         map_height,
                                                         map_width)

        semmap_scores = self.decoder(memory)
        semmap_scores = semmap_scores.squeeze(0)

        observed_masks = observed_masks.reshape(map_height, map_width)
        height_map = height_map.reshape(map_height, map_width)

        return semmap_scores, observed_masks, height_map


class Decoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):
        super(Decoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(inplace=True),
                                   )

        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                       )

    def forward(self, memory):
        l1 = self.layer(memory)
        out_obj = self.obj_layer(l1)
        return out_obj