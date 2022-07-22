import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from Backbone.segformer import Segformer

# from mmcv.cnn import ConvModule


normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])


class Trans4map(nn.Module):
    def __init__(self, cfg, device):
        super(Trans4map, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']
        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # cpu
        # self.device_mem = torch.device('cuda')  # cpu

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

        self.encoder = Segformer()
        self.fuse = nn.Conv2d(mem_feat_dim*2, mem_feat_dim, 1, 1, 0)
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

    def memory_update(self, features, proj_indices, masks_inliers):

        features = features.float() # torch.Size([1, 20, 64, 120, 160])

        N, T, C, H, W = features.shape


        if self.mem_update == 'lstm':
            state = (torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
            state_r = torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
        elif self.mem_update == 'replace':
            state = torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((N, 250, 250), dtype=torch.bool, device=self.device)

        for t in range(T):

            feature = features[:, t, :, :, :].to(self.device)
            mask_inliers = masks_inliers[:, t, :, :]  # torch.Size([1, 480, 640])
            proj_index = proj_indices[:, t, :]  # torch.Size([1, 62500])
            feature_r = features[:, T-t-1, :, :, :].to(self.device)
            mask_inliers_r = masks_inliers[:, T-t-1, :, :]  # torch.Size([1, 480, 640])
            proj_index_r = proj_indices[:, T-t-1, :]  # torch.Size([1, 62500])

            if self.ego_downsample:
                mask_inliers = mask_inliers[:, ::4, ::4]

            m = (proj_index >= 0)  # -- (N, 250*250)
            m_r = (proj_index_r >= 0)  # -- (N, 250*250)

            if N > 1:
                batch_offset = torch.zeros(N, device=self.device)
                batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
                batch_offset = batch_offset.unsqueeze(1).repeat(1, 250 * 250).long()

                proj_index += batch_offset

            if m.any():
                feature = F.interpolate(feature, size=(480, 640), mode="bilinear", align_corners=True)
                if self.ego_downsample:
                    feature = feature[:, :, ::4, ::4]

                feature = feature.permute(0, 2, 3, 1)  # -- (N,H,W,512) # torch.Size([1, 480, 640, 64])

                feature = feature[mask_inliers, :] # torch.Size([90058, 64])

                tmp_memory = feature[proj_index[m], :] # torch.Size([15674, 64])

                tmp_top_down_mask = m.view(-1) # torch.Size([62500])

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][tmp_top_down_mask, :].to(self.device),
                                 state[1][tmp_top_down_mask, :].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][tmp_top_down_mask, :] = tmp_state[0].to(self.device_mem)
                    state[1][tmp_top_down_mask, :] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[tmp_top_down_mask, :].to(self.device) # state: torch.Size([62500, 256])

                    tmp_state = self.rnn(tmp_memory, tmp_state)  # tmp_state: torch.Size([15674, 256])

                    state[tmp_top_down_mask, :] = tmp_state.to(self.device_mem)

                elif self.mem_update == 'replace':
                    tmp_memory = self.linlayer(tmp_memory)
                    state[tmp_top_down_mask, :] = tmp_memory.to(self.device_mem)

                else:
                    raise NotImplementedError

                observed_masks += m.reshape(N, 250, 250) # torch.Size([1, 250, 250])

                del tmp_memory
            if m_r.any():
                feature_r = F.interpolate(feature_r, size=(480, 640), mode="bilinear", align_corners=True) # torch.Size([1, 64, 480, 640])

                feature_r = feature_r.permute(0, 2, 3, 1)  # -- (N,H,W,512) # torch.Size([1, 480, 640, 64])
                feature_r = feature_r[mask_inliers_r, :] # torch.Size([90058, 64])
                tmp_memory_r = feature_r[proj_index_r[m_r], :] # torch.Size([15674, 64])
                tmp_top_down_mask_r = m_r.view(-1) # torch.Size([62500]), 15674==True

                tmp_state_r = state_r[tmp_top_down_mask_r, :].to(self.device) # state: torch.Size([62500, 256])
                tmp_state_r = self.rnn_r(tmp_memory_r, tmp_state_r)  # tmp_state: torch.Size([15674, 256])
                state_r[tmp_top_down_mask_r, :] = tmp_state_r.to(self.device_mem)
                del tmp_memory_r
            del feature, feature_r

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            # memory = state
            memory = torch.cat((state, state_r), dim=-1)
        elif self.mem_update == 'replace':
            memory = state

        memory = memory.view(N, 250, 250, self.mem_feat_dim*2) # torch.Size([1, 250, 250, 256])

        memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
        memory = self.fuse(memory)
        memory = memory.to(self.device)
        return memory, observed_masks

    # def forward(self, features, proj_indices, masks_inliers):
    def forward(self, rgb, proj_indices, masks_inliers):
        features = self.encoder(rgb) # torch.Size([1, 20, 3, 480, 640])
        features = features.unsqueeze(0) # torch.Size([1, 20, 64, 120, 160])
        # predictions = F.interpolate(predictions, size=(480,640), mode="bilinear", align_corners=True)
        memory, observed_masks = self.memory_update(features,
                                                    proj_indices,
                                                    masks_inliers)
        semmap = self.decoder(memory)
        return semmap, observed_masks


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
