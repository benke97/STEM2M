
import torch
import torch.nn as nn

from . import functional as F

__all__ = ['BallQuery']


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(self, points_coords, centers_coords, temb, points_features=None):
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)

        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features, F.grouping(temb, neighbor_indices)

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')


"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # Import the autocast context manager

from . import functional as F

__all__ = ['BallQuery']


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(self, points_coords, centers_coords, temb, points_features=None):
        # The custom functions F.ball_query and F.grouping expect float32 tensors.
        # We will wrap the entire operation in a disabled autocast context to
        # ensure all inputs are handled as float32, preventing crashes with fp16.
        with autocast(enabled=False):
            # Ensure input tensors are float32 and contiguous
            points_coords_fp32 = points_coords.float().contiguous()
            centers_coords_fp32 = centers_coords.float().contiguous()
            temb_fp32 = temb.float()
            
            # The custom C++/CUDA backend functions for ball_query and grouping
            # require float32 inputs.
            neighbor_indices = F.ball_query(centers_coords_fp32, points_coords_fp32, self.radius, self.num_neighbors)
            neighbor_coordinates = F.grouping(points_coords_fp32, neighbor_indices)
            neighbor_coordinates = neighbor_coordinates - centers_coords_fp32.unsqueeze(-1)

            if points_features is None:
                assert self.include_coordinates, 'No Features For Grouping'
                neighbor_features = neighbor_coordinates
            else:
                # Ensure points_features is also float32 before grouping
                points_features_fp32 = points_features.float()
                neighbor_features = F.grouping(points_features_fp32, neighbor_indices)
                if self.include_coordinates:
                    neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
            
            # The final grouping also needs a float32 tensor
            grouped_temb = F.grouping(temb_fp32, neighbor_indices)

        # PyTorch's outer autocast context (from Accelerate) will automatically
        # cast the outputs `neighbor_features` and `grouped_temb` back to fp16
        # for the subsequent layers of the network.
        return neighbor_features, grouped_temb

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')
"""