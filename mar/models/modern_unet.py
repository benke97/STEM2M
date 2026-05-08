# src/mar/models/modern_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Paste your ModernUNet class definition here ---
# Example structure (replace with your actual code):
class DoubleConv(nn.Module):
    # ... your DoubleConv implementation ...
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels), # Use GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    # ... your Down block ...
    def __init__(self, in_channels, out_channels, num_groups=8):
      super().__init__()
      self.maxpool_conv = nn.Sequential(
          nn.MaxPool2d(2),
          DoubleConv(in_channels, out_channels, num_groups=num_groups)
      )
    def forward(self, x):
      return self.maxpool_conv(x)


class Up(nn.Module):
   # ... your Up block ...
    def __init__(self, in_channels, out_channels, bilinear=True, num_groups=8):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, num_groups=num_groups) # Adjust DoubleConv if needed
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, num_groups=num_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    # ... your OutConv block ...
     def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

     def forward(self, x):
        return self.conv(x)


class ModernUNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_features=64, num_groups=8, bilinear=True):
        super(ModernUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        f = init_features

        self.inc = DoubleConv(n_channels, f, num_groups=num_groups)
        self.down1 = Down(f, f*2, num_groups=num_groups)
        self.down2 = Down(f*2, f*4, num_groups=num_groups)
        self.down3 = Down(f*4, f*8, num_groups=num_groups)
        factor = 2 if bilinear else 1
        self.down4 = Down(f*8, f*16 // factor, num_groups=num_groups) # Corrected based on Up block needs
        self.up1 = Up(f*16, f*8 // factor, bilinear, num_groups=num_groups) # Corrected channel sizes
        self.up2 = Up(f*8, f*4 // factor, bilinear, num_groups=num_groups)
        self.up3 = Up(f*4, f*2 // factor, bilinear, num_groups=num_groups)
        self.up4 = Up(f*2, f, bilinear, num_groups=num_groups)
        self.outc = OutConv(f, n_classes)

        # Optional: Add a head for auxiliary tasks if needed, e.g., classification
        # self.aux_head = nn.Sequential(...)

    def forward(self, x): # Removed return_encoder_features argument
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        logits = self.outc(out) # Segmentation map

        encoder_features = (x5, x4, x3, x2, x1)
        #sigmoid logits
        logits = torch.sigmoid(logits)
        # Return both segmentation map (logits) and encoder features
        return logits, encoder_features