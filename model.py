# ============================================================
#  CN-Detect: Model Architecture
#  EfficientNet-B4 + Swin-Tiny + Cross-Attention CAFS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except ImportError:
    import os; os.system("pip install timm -q")
    import timm


class CrossAttentionFS(nn.Module):
    def __init__(self, eff_dim=1792, swin_dim=768,
                 proj_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.eff_proj    = nn.Linear(eff_dim, proj_dim)
        self.swin_proj   = nn.Linear(swin_dim, proj_dim)
        self.eff_to_swin = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.swin_to_eff = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_e = nn.LayerNorm(proj_dim)
        self.norm_s = nn.LayerNorm(proj_dim)
        self.gate_e = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 4), nn.ReLU(),
            nn.Linear(proj_dim // 4, proj_dim), nn.Sigmoid())
        self.gate_s = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 4), nn.ReLU(),
            nn.Linear(proj_dim // 4, proj_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, eff_feat, swin_feat):
        e = self.eff_proj(eff_feat).unsqueeze(1)
        s = self.swin_proj(swin_feat).unsqueeze(1)
        e_att, _ = self.eff_to_swin(query=e, key=s, value=s)
        e_out = self.norm_e(e + e_att).squeeze(1)
        e_out = e_out * self.gate_e(e_out)
        s_att, _ = self.swin_to_eff(query=s, key=e, value=e)
        s_out = self.norm_s(s + s_att).squeeze(1)
        s_out = s_out * self.gate_s(s_out)
        return self.fusion(torch.cat([e_out, s_out], dim=1))


class CNDetectEffB4Swin(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        b4 = models.efficientnet_b4(weights=None)
        self.eff_branch  = nn.Sequential(*list(b4.children())[:-1])
        self.swin_branch = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False, num_classes=0)
        swin_dim = self.swin_branch.num_features
        with torch.no_grad():
            dummy   = torch.zeros(1, 3, 256, 256)
            eff_dim = self.eff_branch(dummy).flatten(1).size(1)
        self.cross_attn_fs = CrossAttentionFS(
            eff_dim=eff_dim, swin_dim=swin_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):
        x224     = F.interpolate(x, size=(224, 224),
                                  mode='bilinear', align_corners=False)
        eff_feat  = self.eff_branch(x).flatten(1)
        swin_feat = self.swin_branch(x224)
        return self.classifier(self.cross_attn_fs(eff_feat, swin_feat))


def load_model(pth_path, device="cpu"):
    """Load model from checkpoint — works with both old and new format"""
    checkpoint = torch.load(pth_path, map_location=device)

    model = CNDetectEffB4Swin(num_classes=4)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        classes  = checkpoint.get("classes",
                   ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"])
        img_size = checkpoint.get("model_config", {}).get("img_size", 256)
    else:
        # old format — state dict only
        model.load_state_dict(checkpoint)
        classes  = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
        img_size = 256

    model.eval()
    return model, classes, img_size