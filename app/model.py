# Shared architecture: used by BOTH train.py and app.py (keeps them identical)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v).reshape(B, C, H, W)
        out = self.norm(self.out_proj(out))
        return x + self.gamma * out


class SlimMHAFusion(nn.Module):
    def __init__(self, in_channels=1280, reduced=320, num_heads=4):
        super().__init__()
        assert reduced % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = reduced // num_heads
        self.reduced = reduced
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False), nn.BatchNorm2d(reduced))
        self.query = nn.Conv2d(reduced, reduced, 1)
        self.key = nn.Conv2d(reduced, reduced, 1)
        self.value = nn.Conv2d(reduced, reduced, 1)
        self.expand = nn.Sequential(
            nn.Conv2d(reduced, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        r = self.reduce(x)
        q = self.query(r).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(r).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(r).view(B, self.num_heads, self.head_dim, H * W)
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v).reshape(B, self.reduced, H, W)
        return x + self.gamma * self.expand(out)


class EfficientNetB0_MHA(nn.Module):
    def __init__(self, num_classes, pretrained=True, reduced=None, num_heads=4):
        super().__init__()
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = backbone.features
        self.mha = (MultiHeadAttentionFusion(1280, num_heads) if reduced is None
                    else SlimMHAFusion(1280, reduced, num_heads))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, num_classes))

    def forward(self, x):
        x = self.mha(self.features(x))
        return self.classifier(torch.flatten(self.avgpool(x), 1))


def create_teacher(num_classes):
    m = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    m.classifier = nn.Sequential(nn.Dropout(p=0.3),
                                 nn.Linear(m.classifier[1].in_features, num_classes))
    return m


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_student(path, device="cpu", reduced=320, heads=4):
    """Load a student checkpoint saved by train.py. Returns (model, class_names)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    m = EfficientNetB0_MHA(len(class_names), pretrained=False, reduced=reduced, num_heads=heads)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.float().to(device).eval(), class_names