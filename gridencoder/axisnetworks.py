import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MultiTriplane(nn.Module):
    def __init__(self, num_objs, input_dim=3, output_dim=1, noise_val = None, device = 'cuda'):
        super().__init__()
        self.device = device
        self.num_objs = num_objs
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 128, 128)*0.001) for _ in range(3*num_objs)])
        self.noise_val = noise_val
        # Use this if you want a PE
        self.net = nn.Sequential(
            FourierFeatureTransform(32, 64, scale=1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, output_dim),
        )

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, obj_idx, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[3*obj_idx+0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[3*obj_idx+1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[3*obj_idx+2])
        
        #if self.noise_val != None:
        #    xy_embed = xy_embed + self.noise_val*torch.empty(xy_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    yz_embed = yz_embed + self.noise_val*torch.empty(yz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    xz_embed = xz_embed + self.noise_val*torch.empty(xz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)

                
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        if self.noise_val != None and self.training:
            features = features + self.noise_val*torch.empty(features.shape).normal_(mean = 0, std = 0.5).to(self.device)
        return self.net(features)
    
    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l/self.num_objs
    
    def l2reg(self):
        l = 0
        for embed in self.embeddings:
            l += (embed**2).sum()**0.5
        return l/self.num_objs
    
class MultiScaleTriplane(nn.Module):
    def __init__(self, input_dim=3, n_scales=3, channel=64, iteration=0, is_training=True, grid_size=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = channel * n_scales
        self.n_scales = n_scales

        # self.embeddings = nn.ModuleList([
        #     nn.ParameterList([
        #         nn.Parameter(torch.randn(1, channel, 256 // (2 ** i), 256 // (2 ** i)) * 0.001)
        #         for _ in range(3)
        #     ])
        #     for i in range(n_scales)
        # ])
        self.plane_x1 = nn.Parameter(torch.randn(1, channel, grid_size, grid_size) * 0.001)
        self.plane_y1 = nn.Parameter(torch.randn(1, channel, grid_size, grid_size) * 0.001)
        self.plane_z1 = nn.Parameter(torch.randn(1, channel, grid_size, grid_size) * 0.001)

        self.embeddings = nn.ModuleList([
            nn.ParameterList([self.plane_x1, self.plane_y1, self.plane_z1]),
            nn.ParameterList([F.avg_pool2d(self.plane_x1, kernel_size=3, stride=2, padding=1),
                              F.avg_pool2d(self.plane_y1, kernel_size=3, stride=2, padding=1),
                              F.avg_pool2d(self.plane_z1, kernel_size=3, stride=2, padding=1)]),
            nn.ParameterList([F.avg_pool2d(F.avg_pool2d(self.plane_x1, kernel_size=3, stride=2, padding=1), kernel_size=3, stride=2, padding=1),
                              F.avg_pool2d(F.avg_pool2d(self.plane_y1, kernel_size=3, stride=2, padding=1), kernel_size=3, stride=2, padding=1),
                              F.avg_pool2d(F.avg_pool2d(self.plane_z1, kernel_size=3, stride=2, padding=1), kernel_size=3, stride=2, padding=1)])
        ])

        # Define the rest of the network (self.net) as before
    
    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bicubic', padding_mode='border', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features
        
    def forward(self, coordinates, debug=False, bound=1, iteration=0, is_training=True, channel=64):
        coordinates = (coordinates + bound) / (2 * bound)
        coordinates = coordinates.unsqueeze(0)
        
        combined_features = torch.zeros(coordinates.shape[0], coordinates.shape[1], self.output_dim, device=coordinates.device)

        for scale_idx in range(self.n_scales):
            xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[scale_idx][0])
            yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[scale_idx][1])
            xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[scale_idx][2])

            combined_features[..., scale_idx*channel:(scale_idx+1)*channel] = xy_embed.add_(yz_embed).add_(xz_embed)


        return combined_features[0]
        # coordinates = (coordinates + bound) / (2 * bound)
        # coordinates = coordinates.unsqueeze(0)

        # features_list = []
        # for scale_idx in range(self.n_scales):
        #     xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[scale_idx][0])
        #     yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[scale_idx][1])
        #     xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[scale_idx][2])

        #     xy_embed.add_(yz_embed).add_(xz_embed)
        #     # features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        #     features_list.append(xy_embed)

        # # CONCAT
        # combined_features = torch.cat(features_list, dim=-1)
        # del features_list
        # return combined_features[0]
        # Combine features from different scales

        # ADD
        # combined_features = features_list[0]
        # for feature in features_list[1:]:
        #     combined_features += feature
    
class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out + F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, initial_scale=1):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * initial_scale, requires_grad=False)
        # self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        # B, N, C = x.shape
        # x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1) * 2 * np.pi
        # x = 2 * np.pi * x
        B, C = x.shape
        x = x @ self._B * 2 * np.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MultiScaleTriplane_Pooling(nn.Module):
    def __init__(self, input_dim=3, n_scales=4, channel=32, grid_size=256, iteration=0, is_training=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = channel
        self.n_scales = n_scales
        self.iteration = iteration
        self.vector_1 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size, 1) * 1e-3) for _ in range(3)])
        self.plane_2 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size // 2, grid_size // 2) * 3e-4) for _ in range(3)])
        self.plane_3 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size // 4, grid_size // 4) * 1.5e-4) for _ in range(3)])
        self.plane_4 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size // 8, grid_size // 8) * 7.5e-5) for _ in range(3)])
        self.net1 = nn.Sequential(
            FourierFeatureTransform(channel, channel // 2, initial_scale=0.0075),
        )
        # self.net2 = nn.Sequential(FourierFeatureTransform(channel, channel, initial_scale=0.1))
    def sample_plane(self, coords2d, plane, is_training):
        assert len(coords2d.shape) == 3, coords2d.shape
        # Generate Gaussian noise with the same shape as coords2d
        # if self.is_training and self.iteration < 3000:
        if is_training:
            coords2d = coords2d + torch.normal(mean=0, std=0.005, size=coords2d.shape).to(coords2d.device)
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.view(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bicubic', padding_mode='border', align_corners=True)
        #mode bicubic padding_mode
        del coords2d
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.view(N, C, H*W).permute(0, 2, 1)
        return sampled_features
    
    def sample_grid(self, coords3d, grid, is_training):
        assert len(coords3d.shape) == 3, coords3d.shape
        if is_training:
            coords3d = coords3d + torch.normal(mean=0, std=0.005, size=coords3d.shape).to(coords3d.device)

        sampled_features = F.grid_sample(grid, coords3d.view(coords3d.shape[0], 1, -1, 1, coords3d.shape[-1]), 
                                         mode='bilinear', padding_mode='border', align_corners=True)
        N, C, _, H, W = sampled_features.shape
        sampled_features = sampled_features.view(N, C, H*W).permute(0, 2, 1)
        return sampled_features
    
    def sample_vector(self, coords1d, vector, is_training):
        assert len(coords1d.shape) == 3, coords1d.shape
        # coords1d = F.pad(coords1d, (0, coords1d.shape[-1]))
        coords1d = torch.stack([-torch.ones_like(coords1d), coords1d], dim=-1)
        if is_training:
            coords1d = coords1d + torch.normal(mean=0, std=0.005, size=coords1d.shape).to(coords1d.device)
        # coords1d = coords1d.view(coords1d.shape[0], 1, -1, coords1d.shape[-1])
        sampled_features = F.grid_sample(vector, coords1d.view(coords1d.shape[0], 1, -1, coords1d.shape[-1]), 
                                         mode='bicubic', padding_mode='border', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.view(N, C, H*W).permute(0, 2, 1)

        return sampled_features
    
    def gridsample1d_by2d(self, input, grid, padding_mode, align_corners):
        shape = grid.shape
        input = input.unsqueeze(-1)  # batch_size * C * L_in * 1
        grid = grid.unsqueeze(1)  # batch_size * 1 * L_out
        grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
        z = F.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
        C = input.shape[1]
        out_shape = [shape[0], C, shape[1]]
        z = z.view(*out_shape)  # batch_size * C * L_out
        return z

    def forward(self, coordinates, debug=False, bound=1, iteration=0, is_training=True):
        # coordinates = (coordinates + bound) / (2 * bound)
        coordinates = coordinates.unsqueeze(0)
        # Update the iteration attribute
        self.iteration = iteration

        plane_x4 = self.plane_4[0].detach() if iteration > 3000 else self.plane_4[0]
        plane_y4 = self.plane_4[1].detach() if iteration > 3000 else self.plane_4[1]
        plane_z4 = self.plane_4[2].detach() if iteration > 3000 else self.plane_4[2]

        xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x4, is_training)
        xy_embed.add_(self.sample_plane(coordinates[..., 1:3], plane_y4, is_training))
        xy_embed.add_(self.sample_plane(coordinates[..., :3:2], plane_z4, is_training))        
        del plane_x4, plane_y4, plane_z4
        
        if iteration > 3000:
            plane_x3 = self.plane_3[0].detach() if iteration > 4000 else self.plane_3[0]
            plane_y3 = self.plane_3[1].detach() if iteration > 4000 else self.plane_3[1]
            plane_z3 = self.plane_3[2].detach() if iteration > 4000 else self.plane_3[2]
            xy_embed.add_(0.7 * self.sample_plane(coordinates[..., 0:2], plane_x3, is_training))
            xy_embed.add_(0.7 * self.sample_plane(coordinates[..., 1:3], plane_y3, is_training))
            xy_embed.add_(0.7 * self.sample_plane(coordinates[..., :3:2], plane_z3, is_training))
            del plane_x3, plane_y3, plane_z3
    
        if iteration > 4000:
            plane_x2 = self.plane_2[0].detach() if iteration > 5000 else self.plane_2[0]
            plane_y2 = self.plane_2[1].detach() if iteration > 5000 else self.plane_2[1]
            plane_z2 = self.plane_2[2].detach() if iteration > 5000 else self.plane_2[2]
            xy_embed.add_(0.5 * self.sample_plane(coordinates[..., 0:2], plane_x2, is_training))
            xy_embed.add_(0.5 * self.sample_plane(coordinates[..., 1:3], plane_y2, is_training))
            xy_embed.add_(0.5 * self.sample_plane(coordinates[..., :3:2], plane_z2, is_training))
            del plane_x2, plane_y2, plane_z2

        if iteration > 5000:
            xy_embed.add_(0.3 * self.sample_vector(coordinates[..., 0:1], self.vector_1[0], is_training))
            xy_embed.add_(0.3 * self.sample_vector(coordinates[..., 1:2], self.vector_1[1], is_training))
            xy_embed.add_(0.3 * self.sample_vector(coordinates[..., 2:3], self.vector_1[2], is_training))

        return self.net1(xy_embed[0])

    def tvreg(self, global_step):
        l = 0
        if global_step <= 3000:
            for embed in self.plane_4:
                l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
                l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        elif global_step <= 4000:
            for embed in self.plane_3:
                l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
                l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        elif global_step <= 5000:
            for embed in self.plane_2:
                l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
                l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        else:
            for embed in self.vector_1:
                l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
                l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l
    
    def l2reg(self, global_step):
        l = 0

        if global_step <= 3000:
            for embed in self.plane_4:
                l += (embed**2).sum()**0.5
        elif global_step <= 4000:
            for embed in self.plane_3:
                l += (embed**2).sum()**0.5
        elif global_step <= 5000:
            for embed in self.plane_2:
                l += (embed**2).sum()**0.5
        else:
            for embed in self.vector_1:
                l += (embed**2).sum()**0.5
        return l