import torch
import torch.nn as nn
import torchvision.models as models

# Your original ResNet18SizeEstimator class (required to define the backbone structure)
class ResNet18SizeEstimator(nn.Module):
    """ResNet-18-based network for estimating nanoparticle sizes from images."""
    def __init__(self, input_nc, output_nc=1):
        """
        Args:
            input_nc (int): Number of input image channels.
            output_nc (int): Number of output channels (1 for regression).
        """
        super(ResNet18SizeEstimator, self).__init__()
        self.input_nc = input_nc
        resnet18 = models.resnet18(weights=None) # Or specific weights if preloaded in torchvision
        resnet18.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1]) # Backbone
        self.fc = nn.Linear(512, output_nc) # Original FC, not used by the feature extractor

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1)
        return x

class ResNet18AdaptivePoolFeatureExtractor(nn.Module):
    """
    Extracts features from a pretrained ResNet18SizeEstimator's backbone
    and uses Adaptive Average Pooling for fixed dimensionality reduction.
    The backbone is frozen and not trained further.
    """
    def __init__(self, input_nc, output_feature_dim=128, pretrained_estimator_path=None):
        super(ResNet18AdaptivePoolFeatureExtractor, self).__init__()
        self.input_nc = input_nc
        self.output_feature_dim = output_feature_dim
        self.input_feature_dim_from_backbone = 512 # ResNet-18 output before original FC

        # 1. Instantiate the base architecture to get the backbone
        # The output_nc for the original model doesn't matter here.
        temp_estimator = ResNet18SizeEstimator(input_nc=self.input_nc, output_nc=1)
        self.resnet_backbone = temp_estimator.resnet18
        # The resnet_backbone already has its first conv layer modified for input_nc by ResNet18SizeEstimator

        # 2. Load pretrained weights for the backbone if path is provided
        if pretrained_estimator_path:
            self.load_pretrained_backbone_weights(pretrained_estimator_path)
        else:
            print("Warning: No pretrained_estimator_path provided. Backbone may be randomly initialized.")

        # 3. Freeze the backbone weights (make them non-trainable)
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False

        # 4. Define the Adaptive Average Pooling layer for dimensionality reduction
        # Input to AdaptiveAvgPool1d is (N, C_in, L_in)
        # Our features from backbone are (batch_size, 512). We'll treat 512 as L_in, and C_in=1.
        self.adaptive_pooler = nn.AdaptiveAvgPool1d(self.output_feature_dim)

    def load_pretrained_backbone_weights(self, path):
        """
        Loads weights from a pretrained ResNet18SizeEstimator checkpoint
        ONLY for the 'resnet18' backbone part.
        """
        try:
            # Load the full state dict of the original ResNet18SizeEstimator
            full_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

            # Create a new state dict containing only the weights for the resnet18 backbone
            backbone_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('resnet18.'):
                    # Remove 'resnet18.' prefix to match keys in self.resnet_backbone
                    new_key = key.replace('resnet18.', '', 1)
                    backbone_state_dict[new_key] = value

            # Load the filtered state dict into the backbone
            missing_keys, unexpected_keys = self.resnet_backbone.load_state_dict(backbone_state_dict, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys when loading into resnet_backbone: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys found in state_dict for resnet_backbone: {unexpected_keys}")
            print(f"Successfully loaded pretrained weights for resnet_backbone from {path}")

        except FileNotFoundError:
            print(f"Error: Pretrained weights file not found at {path}. Backbone remains as initialized.")
        except Exception as e:
            print(f"Error loading pretrained weights for backbone: {e}. Backbone remains as initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass to extract features and reduce dimensionality.
        Args:
            x: Input image tensor of shape (batch_size, input_nc, H, W).
        Returns:
            Tensor of shape (batch_size, output_feature_dim).
        """
        # Ensure backbone is in evaluation mode as it's frozen and for layers like BatchNorm
        self.resnet_backbone.eval()

        with torch.no_grad(): # Gradients are not needed for the frozen backbone
            # x shape: (batch_size, input_nc, H, W)
            features_backbone = self.resnet_backbone(x)
            # Output shape of resnet_backbone (after removing original FC): (batch_size, 512, 1, 1) for typical inputs
            # Flatten the features: (batch_size, 512)
            features_flattened = features_backbone.view(features_backbone.size(0), -1)

            # Reshape for AdaptiveAvgPool1d: (batch_size, C_in, L_in) -> (batch_size, 1, 512)
            # We treat the 512 features as the length dimension to be pooled over, with 1 input channel.
            features_reshaped = features_flattened.unsqueeze(1)

            # Apply adaptive average pooling to reduce from 512 to output_feature_dim
            # Output shape: (batch_size, 1, output_feature_dim)
            projected_features = self.adaptive_pooler(features_reshaped)

            # Squeeze to remove the channel dimension: (batch_size, output_feature_dim)
            projected_features = projected_features.squeeze(1)

        return projected_features