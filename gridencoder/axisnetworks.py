import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init
        
class Sine(nn.Module):
    """Sine Activation Function."""
    
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Siren(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            Sine(),
            nn.Linear(512, 512),
            Sine(),
            nn.Linear(512, 512),
            Sine(),
            nn.Linear(512, output_dim)
        )
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)
    def forward(self, input, debug=False):
        return self.net(input)

def linearInterpolate(coordinates, values, axis=0):
    coordinates = torch.clamp(coordinates, -1, 0.999)
    axis_coords = (0.5*coordinates + 0.5) * (values.shape[axis] - 1)
    axis_indices_0 = torch.floor(axis_coords).long()
    axis_indices_1 = axis_indices_0 + 1
    
#     v_0 = values[axis_indices_0]
#     v_1 = values[axis_indices_1]
    v_0 = torch.index_select(values, 0, axis_indices_0.squeeze())
    v_1 = torch.index_select(values, 0, axis_indices_1.squeeze())

    w = (axis_coords - axis_indices_0).unsqueeze(-1)
    
#     print(w.shape, v_0.shape, v_1.shape)

    return (1 - w) * v_0 + w * v_1

def fastCollect(coordinates, values, axis=0):
#     return torch.randn(coordinates.shape[-1], values.shape[-1], device=coordinates.device)
    
    coordinates = torch.clamp(coordinates, -1, 0.999)
    axis_coords = (0.5*coordinates + 0.5) * (values.shape[axis] - 1)
    axis_indices_0 = torch.floor(axis_coords).long()
    
#     return values[axis_indices_0]
#     print(values.shape, axis_indices_0.shape)
    return torch.index_select(values, 0, axis_indices_0.squeeze())


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class AxisNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, axis_resolution=512, embedding_dim=256, num_encoding_functions=1):
        super().__init__()
        
        self.axis_resolution = axis_resolution
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.axis_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(self.axis_resolution, self.embedding_dim, dtype=torch.float)*0.1)
             for _ in range(input_dim)])
        
        print(f"Model with {len(self.axis_embeddings)} axes")
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            Sine(),
            
            nn.Linear(128, 128),
            Sine(),
            
            nn.Linear(128, output_dim)
        )
        
        self.decoder.apply(frequency_init(30))
        self.decoder[0].apply(first_layer_sine_init)
        
    def forward(self, coords, debug=False):    
        assert coords.shape[-1] == len(self.axis_embeddings), f"{coords.shape[-1]}, {len(self.axis_embeddings)}"
        assert (coords.min() >= -1).all() and (coords.max() <= 1).all(), f"{coords.min()} {coords.max()}"
            
        embeddings = [linearInterpolate(coords[..., axis], self.axis_embeddings[axis]) for axis in range(len(self.axis_embeddings))]

        embeddings = torch.prod(torch.stack(embeddings), dim=0)    
        # try max pooling?
                
        return self.decoder(embeddings)
    
class MultiAxisNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, axis_resolution=512, embedding_dim=256, num_encoding_functions=1):
        super().__init__()
        
        self.axis_resolution = axis_resolution
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.axis_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(self.axis_resolution, self.embedding_dim, dtype=torch.float)*0.1)
             for _ in range(input_dim + 2)])
        
        print(f"Model with {len(self.axis_embeddings)} axes")
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            Sine(),
            nn.Linear(256, output_dim)
        )
        
        self.decoder.apply(frequency_init(30))
        self.decoder[0].apply(first_layer_sine_init)
        
    def forward(self, coords, debug=False):
#         coords = self.positional_encoding(coords)
#         coords = torch.cat([coords, (coords[..., 0:1] + coords[..., 1:2])/2], -1)

        coords = torch.clamp(coords, -1, 1)
    
        coords = torch.cat([coords, (coords[..., 0:1] + coords[..., 1:2])/2, (coords[..., 0:1] - coords[..., 1:2])/2], -1)
    
        assert coords.shape[-1] == len(self.axis_embeddings), f"{coords.shape[-1]}, {len(self.axis_embeddings)}"
        assert (coords.min() >= -1).all() and (coords.max() <= 1).all(), f"{coords.min()} {coords.max()}"
            
        embeddings = [linearInterpolate(coords[..., axis], self.axis_embeddings[axis]) for axis in range(len(self.axis_embeddings))]


        embeddings = torch.mean(torch.stack(embeddings), dim=0)    
        # try max pooling?
                
        return self.decoder(embeddings)
    
class VolumeEmbeddingNetwork(nn.Module):
    def __init__(self, embedding_shape=None, input_dim=2, output_dim=3):
        super().__init__()
        
        if input_dim == 2:
            self.embeddings = nn.Parameter(torch.randn(1, *embedding_shape)*0.1)
        elif input_dim == 3:
            self.embeddings = nn.Parameter(torch.randn(1, *embedding_shape)*0.1)

        self.net = nn.Sequential(
            nn.Linear(embedding_shape[0], 128),
            Sine(),
#             nn.LeakyReLU(0.2),
            
            
            nn.Linear(128, 128),
            Sine(),
#             nn.LeakyReLU(0.2),
            
            nn.Linear(128, output_dim),
        )
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)
    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        if n_dims == 2:
            sampled_features = torch.nn.functional.grid_sample(self.embeddings,
                                                               coordinates.reshape(batch_size, 1, -1, n_dims),
                                                               mode='bilinear', padding_mode='zeros', align_corners=True)
            N, C, H, W = sampled_features.shape
            sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        elif n_dims == 3:
            sampled_features = torch.nn.functional.grid_sample(self.embeddings,
                                                               coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                               mode='bilinear', padding_mode='zeros', align_corners=True)
            N, C, H, W, D = sampled_features.shape
            sampled_features = sampled_features.reshape(N, C, H*W*D).permute(0, 2, 1)
        return self.net(sampled_features)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)
        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.
        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).
        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)
        
        
class BarycentricNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, axis_resolution=512, embedding_dim=256, num_axes=8):
        super().__init__()
        
        self.axis_resolution = axis_resolution
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.anchor_locations = nn.Parameter(torch.rand(num_axes, input_dim) * 2 - 1, requires_grad=False)
        
        self.axis_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(self.axis_resolution, self.embedding_dim, dtype=torch.float)*0.1)
             for _ in range(self.anchor_locations.shape[0])])
        
        print(f"Model with {len(self.axis_embeddings)} axes")
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            Sine(),
            nn.Linear(256, output_dim)
        )
        
        self.decoder.apply(frequency_init(30))
        self.decoder[0].apply(first_layer_sine_init)
        
#     def compute_distance_embedding(self, coordinates):
#         distances = []
#         for anchor in self.anchors:
#             distances.append(torch.norm(coordinates - anchor, dim=-1))
#         return torch.stack(distances, -1)
    def compute_distances(self, coordinates):
        coordinates = coordinates.squeeze(0)
        anchors = self.anchor_locations
        distances = torch.norm(coordinates.unsqueeze(1) - anchors.unsqueeze(0).expand(coordinates.shape[0], -1, -1), dim=-1)
        return distances
        
    def forward(self, coords, debug=False):
        coords = self.compute_distances(coords)
        assert coords.shape[-1] == len(self.axis_embeddings)
            
        embeddings = [linearInterpolate(coords[..., axis], self.axis_embeddings[axis]) for axis in range(len(self.axis_embeddings))]


        embeddings = torch.mean(torch.stack(embeddings), dim=0)    
        # try max pooling?
                
        return self.decoder(embeddings)


class PureBarycentricNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, embedding_dim=256, num_anchors=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
#         self.anchor_locations = nn.Parameter(torch.rand(num_anchors, input_dim) * 2 - 1, requires_grad=False)
        dirs = torch.randn(num_anchors, input_dim)
        dirs = dirs/torch.norm(dirs, dim=-1, keepdim=True)
#         self.anchor_locations = nn.Parameter(dirs * 3**0.5, requires_grad=False)
        self.anchor_locations = nn.Parameter(dirs * 2**0.5, requires_grad=False)


        self.anchor_values = nn.Parameter(torch.randn(num_anchors, embedding_dim))
                
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            Sine(),
            nn.Linear(256, output_dim)
        )
        
        self.decoder.apply(frequency_init(30))
        self.decoder[0].apply(first_layer_sine_init)
        
#     def compute_distance_embedding(self, coordinates):
#         distances = []
#         for anchor in self.anchor_locations:
#             distances.append(torch.norm(coordinates - anchor, dim=-1))
#         return torch.stack(distances, -1)
    def compute_distances(self, coordinates):
        anchors = self.anchor_locations
        distances = torch.norm(coordinates.unsqueeze(1) - anchors.unsqueeze(0).expand(coordinates.shape[0], -1, -1), dim=-1)
        return distances
    
    def compute_weights(self, coordinates):
        return (1 - self.compute_distances(coordinates.squeeze())/(12**0.5)).unsqueeze(0)

    def forward(self, coords, debug=False):
        
        weights = self.compute_weights(coords)**2
#         weights = torch.clamp(weights - 0.7, 0, 1)
        assert weights.min() >= 0 and weights.max() <= 1
#         print(weights.min(), weights.max(), weights.mean())
        
#         print(coords.shape, weights.shape, self.anchor_values.shape)
        embeddings = torch.matmul(weights, self.anchor_values)
        return self.decoder(embeddings)




class CartesianPlaneEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super().__init__()
        
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 128, 256, 256)*0.1) for _ in range(3)])

        self.net = nn.Sequential(
            nn.Linear(128, 128),
            Sine(),
            
            nn.Linear(128, 128),
            Sine(),
            
            nn.Linear(128, output_dim),
        )
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[0])
        
#         print(xy_embed.shape, yz_embed.shape, xz_embed.shape)
        
        features = torch.prod(torch.stack([xy_embed, xz_embed, xz_embed]), dim=0) 
        return self.net(features)


class CartesianPlaneEmbeddingNetwork2(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super().__init__()
        
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 256, 256)*0.1) for _ in range(3)])

        self.net = nn.Sequential(
            nn.Linear(32, 64),
            Sine(),
            
            nn.Linear(64, 64),
            Sine(),
            
            nn.Linear(64, output_dim),
        )
        
        self.net.apply(frequency_init(5))
        self.net[0].apply(first_layer_sine_init)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])
        
#         print(xy_embed.shape, yz_embed.shape, xz_embed.shape)
        
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        return self.net(features)



class MiniTriplane(nn.Module):
    def __init__(self, input_dim=3, output_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 128, 128)*0.001) for _ in range(3)])

        # Use this if you want a PE
        self.net = nn.Sequential(
            FourierFeatureTransform(32, 64, scale=1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, output_dim),
        )
        
        # self.net = nn.Sequential(
        #     nn.Linear(32, 128),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(128, output_dim),
        # )


    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False, bound=1):
        # batch_size, n_coords, n_dims = coordinates.shape
        coordinates = (coordinates + bound) / (2 * bound)
        coordinates = coordinates.unsqueeze(0)
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])
                
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        # out = self.net(features)[0]
        return features[0]


    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l
    
    

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
    def __init__(self, input_dim=3, n_scales=3, iteration=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 32 * n_scales
        self.n_scales = n_scales

        self.embeddings = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.randn(1, 32, 256 // (2 ** i), 256 // (2 ** i)) * 0.001)
                for _ in range(3)
            ])
            for i in range(n_scales)
        ])

        # Define the rest of the network (self.net) as before
    
    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features
        
    def forward(self, coordinates, debug=False, bound=1, iteration=0):
        coordinates = (coordinates + bound) / (2 * bound)
        coordinates = coordinates.unsqueeze(0)

        features_list = []
        for scale_idx in range(self.n_scales):
            xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[scale_idx][0])
            yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[scale_idx][1])
            xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[scale_idx][2])

            features = xy_embed.add_(yz_embed).add_(xz_embed)
            # features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
            features_list.append(features)

        # Combine features from different scales

        # ADD
        # combined_features = features_list[0]
        # for feature in features_list[1:]:
        #     combined_features += feature

        # CONCAT
        combined_features = torch.cat(features_list, dim=-1)
        return combined_features[0]
    

# class MultiScaleTriplane_Pooling(nn.Module):
#     def __init__(self, input_dim=3, n_scales=3):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = 32 * n_scales
#         self.n_scales = n_scales

#         self.plane_x1 = nn.Parameter(torch.randn(1, 32, 128, 128) * 0.001)
#         self.plane_x2 = F.avg_pool2d(self.plane_x1, kernel_size=3, stride=2, padding=1)
#         self.plane_x3 = F.avg_pool2d(self.plane_x2, kernel_size=3, stride=2, padding=1)

#         self.plane_y1 = nn.Parameter(torch.randn(1, 32, 128, 128) * 0.001)
#         self.plane_y2 = F.avg_pool2d(self.plane_y1, kernel_size=3, stride=2, padding=1)
#         self.plane_y3 = F.avg_pool2d(self.plane_y2, kernel_size=3, stride=2, padding=1)

#         self.plane_z1 = nn.Parameter(torch.randn(1, 32, 128, 128) * 0.001)
#         self.plane_z2 = F.avg_pool2d(self.plane_z1, kernel_size=3, stride=2, padding=1)
#         self.plane_z3 = F.avg_pool2d(self.plane_z2, kernel_size=3, stride=2, padding=1)
        
#         self.embeddings = nn.ModuleList([
#             nn.ParameterList([
#                 self.plane_x1,
#                 self.plane_y1,
#                 self.plane_z1
#             ]),

#             nn.ParameterList([
#                 self.plane_x2,
#                 self.plane_y2,
#                 self.plane_z2
#             ]),

#             nn.ParameterList([
#                 self.plane_x3,
#                 self.plane_y3,
#                 self.plane_z3
#             ])
#         ])

#         # Define the rest of the network (self.net) as before
    
#     def sample_plane(self, coords2d, plane):
#         assert len(coords2d.shape) == 3, coords2d.shape
#         sampled_features = torch.nn.functional.grid_sample(plane,
#                                                            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
#                                                            mode='bilinear', padding_mode='zeros', align_corners=True)
#         N, C, H, W = sampled_features.shape
#         sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
#         return sampled_features
        
#     def forward(self, coordinates, debug=False, bound=1):
#         coordinates = (coordinates + bound) / (2 * bound)
#         coordinates = coordinates.unsqueeze(0)

#         features_list = []
#         for scale_idx in range(self.n_scales):
#             xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[scale_idx][0])
#             yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[scale_idx][1])
#             xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[scale_idx][2])

#             features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
#             features_list.append(features)

#         # Combine features from different scales

#         # ADD
#         # combined_features = features_list[0]
#         # for feature in features_list[1:]:
#         #     combined_features += feature

#         # CONCAT
#         combined_features = torch.cat(features_list, dim=-1)

#         return combined_features[0]
    
class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out + F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)


class MultiScaleTriplane_Pooling(nn.Module):
    def __init__(self, input_dim=3, n_scales=3, channel=64, grid_size=256, iteration=0, is_training=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = channel 
        self.n_scales = n_scales
        self.res_block = ResidualBlock(channel)
        # self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        # # self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=2, padding=2)
        # self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, stride=4, padding=3)
        # # self.bn = nn.BatchNorm2d(channel)
        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv3.weight)
        self.plane_x1 = nn.Parameter(torch.randn(1, channel, grid_size, grid_size) * 0.001)
        self.plane_y1 = nn.Parameter(torch.randn(1, channel, grid_size, grid_size) * 0.001)
        self.plane_z1 = nn.Parameter(torch.randn(1, channel, grid_size, grid_size) * 0.001)


    def sample_plane(self, coords2d, plane, is_training):
        assert len(coords2d.shape) == 3, coords2d.shape
        # Generate Gaussian noise with the same shape as coords2d
        # if self.is_training and self.iteration < 3000:
        if is_training:
            noise = torch.normal(mean=0, std=0.01, size=coords2d.shape).to(coords2d.device)
            coords2d = coords2d + noise
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bicubic', padding_mode='border', align_corners=True)
        #mode bicubic padding_mode
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features
        
    def forward(self, coordinates, debug=False, bound=1, iteration=0, is_training=True):
        coordinates = (coordinates + bound) / (2 * bound)
        coordinates = coordinates.unsqueeze(0)

        plane_x = self.plane_x1
        plane_y = self.plane_y1
        plane_z = self.plane_z1

        # # CONV
        # if iteration < 3000:
        #     conv_fn = self.conv3
        # else:
        #     conv_fn = self.conv1

        # plane_x = F.relu(conv_fn(self.plane_x1))
        # plane_y = F.relu(conv_fn(self.plane_y1))
        # plane_z = F.relu(conv_fn(self.plane_z1))
        plane_x = self.res_block(plane_x)
        plane_y = self.res_block(plane_y)
        plane_z = self.res_block(plane_z)



        xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x, is_training)
        yz_embed = self.sample_plane(coordinates[..., 1:3], plane_y, is_training)
        xz_embed = self.sample_plane(coordinates[..., :3:2], plane_z, is_training)

        features = xy_embed.add_(yz_embed).add_(xz_embed)

        del plane_x, plane_y, plane_z, xy_embed, yz_embed, xz_embed
        return features[0]
        # if iteration < 3000:
        #     n_pooling = self.n_scales - 1
        # else:
        #     n_pooling = self.n_scales - 2

        # for _ in range(n_pooling):
        #     plane_x = F.avg_pool2d(plane_x, kernel_size=3, stride=2, padding=1)
        #     plane_y = F.avg_pool2d(plane_y, kernel_size=3, stride=2, padding=1)
        #     plane_z = F.avg_pool2d(plane_z, kernel_size=3, stride=2, padding=1)

        # xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x, is_training)
        # yz_embed = self.sample_plane(coordinates[..., 1:3], plane_y, is_training)
        # xz_embed = self.sample_plane(coordinates[..., :3:2], plane_z, is_training)

        # features = xy_embed.add_(yz_embed).add_(xz_embed)

        # del plane_x, plane_y, plane_z, xy_embed, yz_embed, xz_embed
        # # low -> mid/ low -> high
        # if self.iteration < 3000:
        #     for _ in range(self.n_scales - 1):
        #         plane_x = F.avg_pool2d(plane_x, kernel_size=3, stride=2, padding=1)
        #         plane_y = F.avg_pool2d(plane_y, kernel_size=3, stride=2, padding=1)
        #         plane_z = F.avg_pool2d(plane_z, kernel_size=3, stride=2, padding=1)
        #     # low resolution
        #     xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x)
        #     yz_embed = self.sample_plane(coordinates[..., 1:3], plane_y)
        #     xz_embed = self.sample_plane(coordinates[..., :3:2], plane_z)
        #     # features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        #     features = xy_embed.add_(yz_embed).add_(xz_embed)

        # # # only low
        # # for _ in range(self.n_scales - 1):
        # #     plane_x = F.avg_pool2d(plane_x, kernel_size=3, stride=2, padding=1)
        # #     plane_y = F.avg_pool2d(plane_y, kernel_size=3, stride=2, padding=1)
        # #     plane_z = F.avg_pool2d(plane_z, kernel_size=3, stride=2, padding=1)
        #     # low resolution

        # else:
        #     # mid resolution
        #     plane_x = F.avg_pool2d(plane_x, kernel_size=3, stride=2, padding=1)
        #     plane_y = F.avg_pool2d(plane_y, kernel_size=3, stride=2, padding=1)
        #     plane_z = F.avg_pool2d(plane_z, kernel_size=3, stride=2, padding=1)
        #     xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x)
        #     yz_embed = self.sample_plane(coordinates[..., 1:3], plane_y)
        #     xz_embed = self.sample_plane(coordinates[..., :3:2], plane_z)
        #     # features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        #     features = xy_embed.add_(yz_embed).add_(xz_embed)
        # del plane_x, plane_y, plane_z, xy_embed, yz_embed, xz_embed

        # xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x)
        # yz_embed = self.sample_plane(coordinates[..., 1:3], plane_y)
        # xz_embed = self.sample_plane(coordinates[..., :3:2], plane_z)
        # # features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        # features = xy_embed.add_(yz_embed).add_(xz_embed)
        # # only mid
        # plane_x = F.avg_pool2d(plane_x, kernel_size=3, stride=2, padding=1)
        # plane_y = F.avg_pool2d(plane_y, kernel_size=3, stride=2, padding=1)
        # plane_z = F.avg_pool2d(plane_z, kernel_size=3, stride=2, padding=1)
        # xy_embed = self.sample_plane(coordinates[..., 0:2], plane_x)
        # yz_embed = self.sample_plane(coordinates[..., 1:3], plane_y)
        # xz_embed = self.sample_plane(coordinates[..., :3:2], plane_z)
        # # features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        # features = xy_embed.add_(yz_embed).add_(xz_embed)


        # xy_embed_high = self.sample_plane(coordinates[..., 0:2], plane_x, is_training)
        # yz_embed_high = self.sample_plane(coordinates[..., 1:3], plane_y, is_training)
        # xz_embed_high = self.sample_plane(coordinates[..., :3:2], plane_z, is_training)
        
        # for _ in range(self.n_scales - 1):
        #     plane_x = F.avg_pool2d(plane_x, kernel_size=3, stride=2, padding=1)
        #     plane_y = F.avg_pool2d(plane_y, kernel_size=3, stride=2, padding=1)
        #     plane_z = F.avg_pool2d(plane_z, kernel_size=3, stride=2, padding=1)
        # #stride conv 1, 2, 4
        # xy_embed_low = self.sample_plane(coordinates[..., 0:2], plane_x, is_training)
        # yz_embed_low = self.sample_plane(coordinates[..., 1:3], plane_y, is_training)
        # xz_embed_low = self.sample_plane(coordinates[..., :3:2], plane_z, is_training)

        # # Gradually increase the weight of high resolution features
        # alpha = min(1.0, iteration / 3000.0)

        # xy_embed_low = ((1 - alpha) * xy_embed_low).add_(alpha * xy_embed_high)
        # yz_embed_low = ((1 - alpha) * yz_embed_low).add_(alpha * yz_embed_high)
        # xz_embed_low = ((1 - alpha) * xz_embed_low).add_(alpha * xz_embed_high)

        # features = xy_embed_low.add_(yz_embed_low).add_(xz_embed_low)

        # del plane_x, plane_y, plane_z, xy_embed_low, yz_embed_low, xz_embed_low, xy_embed_high, yz_embed_high, xz_embed_high





        
    
class CartesianPlaneNonSirenEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super().__init__()
        
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 64, 128, 128)*0.1) for _ in range(3)])

        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim),
        )
        
        self.coord_encoder = nn.Linear(3, 64)
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[0])
        coord_embed = self.coord_encoder(coordinates)
#         print(xy_embed.shape, yz_embed.shape, xz_embed.shape)
        
        features = torch.prod(torch.stack([coord_embed, xy_embed, xz_embed, xz_embed]), dim=0)
        return self.net(features)
    
    
class VolumeConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super().__init__()
        
        if input_dim == 2:
            self.seed = nn.Parameter(torch.randn(1, 128, 8, 8))
            self.feature_generator = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.LeakyReLU(),
                
                nn.Upsample(64),
                
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LeakyReLU(),
                
                nn.Upsample(128),
    
                nn.Conv2d(64, 32, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
            )
        elif input_dim == 3:
            self.seed = nn.Parameter(torch.randn(1, 128, 4, 4, 4))
            self.feature_generator = nn.Sequential(
                nn.Conv3d(128, 64, 3, padding=1),
                nn.LeakyReLU(),
                
                nn.Upsample(16),
                
                nn.Conv3d(64, 64, 3, padding=1),
                nn.LeakyReLU(),
                
                nn.Upsample(32),
                
                nn.Conv3d(64, 64, 3, padding=1),
                nn.LeakyReLU(),
            )

        self.net = nn.Sequential(
            nn.Linear(64, 128),
            Sine(),
            
            nn.Linear(128, 128),
            Sine(),
            
            nn.Linear(128, output_dim),
        )
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)
    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        if n_dims == 2:
            sampled_features = torch.nn.functional.grid_sample(self.embeddings,
                                                               coordinates.reshape(batch_size, 1, -1, n_dims),
                                                               mode='bilinear', padding_mode='zeros', align_corners=True)
            N, C, H, W = sampled_features.shape
            sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        elif n_dims == 3:
            embeddings = self.feature_generator(self.seed)
            sampled_features = torch.nn.functional.grid_sample(embeddings,
                                                               coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                               mode='bilinear', padding_mode='zeros', align_corners=True)
            N, C, H, W, D = sampled_features.shape
            sampled_features = sampled_features.reshape(N, C, H*W*D).permute(0, 2, 1)
        return self.net(sampled_features)
    
    
    
# from gridwarp import GridWarp3D
# class WarpedVolumeNetwork(nn.Module):
#     def __init__(self, embedding_shape=None, output_dim=1):
#         super().__init__()
#         self.embeddings = nn.Parameter(torch.randn(1, *embedding_shape)*0.1)

#         self.net = nn.Sequential(
#             nn.Linear(embedding_shape[0], 128),
#             Sine(),            
#             nn.Linear(128, 128),
#             Sine(),            
#             nn.Linear(128, output_dim),
#         )
#         self.net.apply(frequency_init(30))
#         self.net[0].apply(first_layer_sine_init)
        
#         self.gridwarper = GridWarp3D(16, 0.2)

#     def forward(self, coordinates, debug=False):
#         coordinates = self.gridwarper(coordinates)
        
#         batch_size, n_coords, n_dims = coordinates.shape
#         sampled_features = torch.nn.functional.grid_sample(self.embeddings,
#                                                            coordinates.reshape(batch_size, 1, 1, -1, n_dims),
#                                                            mode='bilinear', padding_mode='zeros', align_corners=True)
#         N, C, H, W, D = sampled_features.shape
#         sampled_features = sampled_features.reshape(N, C, H*W*D).permute(0, 2, 1)
#         return self.net(sampled_features)