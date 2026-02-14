"""
Attention U-Net Model for Oil Spill Segmentation
Encoder-decoder architecture with attention mechanisms for improved feature focusing
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with ReLU activation.
    Standard building block for U-Net architectures.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate mechanism for better feature focusing.
    Helps model focus on relevant regions while suppressing irrelevant ones.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # Gating signal transformation
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Skip connection transformation
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        Args:
            x: Skip connection features
            g: Gating signal from decoder
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for oil spill segmentation.

    Architecture:
    - Encoder: 4 levels with max pooling
    - Bottleneck: Dense feature extraction
    - Decoder: 4 levels with attention gates and skip connections
    - Output: Single channel binary segmentation mask

    Input: Grayscale satellite image [B, 1, 256, 256]
    Output: Binary segmentation mask [B, 1, 256, 256]
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (Upsampling path) with Attention Gates
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input image tensor [B, 1, H, W]

        Returns:
            Output logits [B, 1, H, W]
        """
        # Encoder with skip connections
        e1 = self.enc1(x)           # 64 channels
        p1 = self.pool(e1)          # /2

        e2 = self.enc2(p1)          # 128 channels
        p2 = self.pool(e2)          # /4

        e3 = self.enc3(p2)          # 256 channels
        p3 = self.pool(e3)          # /8

        e4 = self.enc4(p3)          # 512 channels
        p4 = self.pool(e4)          # /16

        # Bottleneck
        b = self.bottleneck(p4)     # 1024 channels

        # Decoder with attention gates
        d4 = self.up4(b)            # Upsample to /8
        e4_att = self.att4(e4, d4)  # Apply attention to skip connection
        d4 = torch.cat([d4, e4_att], dim=1)  # Concatenate
        d4 = self.dec4(d4)          # 512 channels

        d3 = self.up3(d4)           # Upsample to /4
        e3_att = self.att3(e3, d3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)          # 256 channels

        d2 = self.up2(d3)           # Upsample to /2
        e2_att = self.att2(e2, d2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)          # 128 channels

        d1 = self.up1(d2)           # Upsample to original size
        e1_att = self.att1(e1, d1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)          # 64 channels

        # Output (logits, apply sigmoid during inference)
        out = self.out(d1)

        return out


def test_model():
    """Test model with dummy input"""
    print("Testing Attention U-Net model...")

    model = AttentionUNet(in_channels=1, out_channels=1)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 256, 256)
    print(f"\nInput shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Apply sigmoid for probability
    prob = torch.sigmoid(output)
    print(f"Output range (logits): [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output range (probs): [{prob.min():.4f}, {prob.max():.4f}]")

    print("\n✅ Model test passed!")


if __name__ == '__main__':
    test_model()
