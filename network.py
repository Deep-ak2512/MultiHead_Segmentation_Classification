import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_stride1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1):
        super(Residual_stride1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = x + self.layer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=2, stride=2):
        super(Upsample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x


class UNet_Segmentation(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, nc=[64, 128, 256, 512]):
        super(UNet_Segmentation, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.nc = nc
        
        # Encoder - track spatial dimensions: H → H/2 → H/4 → H/8
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.nc[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_1 = nn.Sequential(
            Downsample(self.nc[0], self.nc[1], 3, 2, 1),
            Residual_stride1(self.nc[1], self.nc[1], 3, 1, 1)
        )
        
        self.encoder_2 = nn.Sequential(
            Downsample(self.nc[1], self.nc[2], 3, 2, 1),
            Residual_stride1(self.nc[2], self.nc[2], 3, 1, 1)
        )
        
        self.encoder_3 = nn.Sequential(
            Downsample(self.nc[2], self.nc[3], 3, 2, 1),
            Residual_stride1(self.nc[3], self.nc[3], 3, 1, 1)
        )
        
        self.bottleneck = Residual_stride1(self.nc[3], self.nc[3], 3, 1, 1)
        
        # Decoder - restore spatial dimensions: H/8 → H/4 → H/2 → H
        self.decoder_3 = Upsample(self.nc[3], self.nc[2], 2, 2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(self.nc[2] + self.nc[2], self.nc[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.nc[2]),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_2 = Upsample(self.nc[2], self.nc[1], 2, 2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(self.nc[1] + self.nc[1], self.nc[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.nc[1]),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_1 = Upsample(self.nc[1], self.nc[0], 2, 2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(self.nc[0] + self.nc[0], self.nc[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.nc[0]),
            nn.ReLU(inplace=True)
        )
        
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.nc[0], 1, kernel_size=3, stride=1, padding=1)
        )
        
        #classification head
        self.cls_head = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(self.nc[3],128),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.3),
                                    nn.Linear(128,3)
,                                    )
    
    def forward(self, x):
        # Encoder path with spatial dimensions
        x0 = self.head(x)          # [B, 64, H, W]
        x1 = self.encoder_1(x0)     # [B, 128, H/2, W/2]
        x2 = self.encoder_2(x1)     # [B, 256, H/4, W/4]
        x3 = self.encoder_3(x2)     # [B, 512, H/8, W/8]
        x4 = self.bottleneck(x3)    # [B, 512, H/8, W/8]
        cls_logits = self.cls_head(x4)
        
        # Decoder path with skip connections (concatenation)
        # Upsample from H/8 to H/4, concat with x2 (H/4)
        x3_up = self.decoder_3(x4)              # [B, 256, H/4, W/4]
        x3_up = torch.cat([x3_up, x2], dim=1)   # [B, 512, H/4, W/4]
        x3_up = self.dec3_conv(x3_up)           # [B, 256, H/4, W/4]
        
        # Upsample from H/4 to H/2, concat with x1 (H/2)
        x2_up = self.decoder_2(x3_up)            # [B, 128, H/2, W/2]
        x2_up = torch.cat([x2_up, x1], dim=1)   # [B, 256, H/2, W/2]
        x2_up = self.dec2_conv(x2_up)           # [B, 128, H/2, W/2]
        
        # Upsample from H/2 to H, concat with x0 (H)
        x1_up = self.decoder_1(x2_up)            # [B, 64, H, W]
        x1_up = torch.cat([x1_up, x0], dim=1)   # [B, 128, H, W]
        x1_up = self.dec1_conv(x1_up)           # [B, 64, H, W]
        
        seg_logits = self.last_layer(x1_up)  # [B, 1, H, W]
        
        return seg_logits,cls_logits


if __name__ == "__main__":
    # Test with different input sizes
    print("Testing UNet Segmentation Model...")
    
    test_sizes = [(1, 1, 224, 224), (1, 1, 256, 256), (2, 1, 512, 512)]
    
    for inp_shape in test_sizes:
        inp = torch.rand(*inp_shape)
        model = UNet_Segmentation(in_channels=1, n_classes=3)
        out_seg,out_cls = model(inp)
        print(f"\nInput shape: {inp_shape}")
        print(f"Output shape: {out_seg.shape}")
        print(f"Output shape: {out_cls.shape}")
        print(f"Output range: [{out_seg.min():.3f}, {out_seg.max():.3f}]")
        print(f"Output range: [{out_cls.min():.3f}, {out_cls.max():.3f}]")
