# ============================================================
#  Convert old .pth (state_dict only) to complete checkpoint
#  Run this ONCE to convert your existing model
#  Input:  cn_detect_effb4_swin_best.pth (old format)
#  Output: cn_detect_webapp.pth (complete format)
# ============================================================

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:
    import os; os.system("pip install timm -q")
    import timm

# ─────────────────────────────────────────────────────────────
# SAME ARCHITECTURE AS TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────
class CrossAttentionFS(nn.Module):
    def __init__(self, eff_dim=1792, swin_dim=768,
                 proj_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.eff_proj    = nn.Linear(eff_dim, proj_dim)
        self.swin_proj   = nn.Linear(swin_dim, proj_dim)
        self.eff_to_swin = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.swin_to_eff = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_e = nn.LayerNorm(proj_dim)
        self.norm_s = nn.LayerNorm(proj_dim)
        self.gate_e = nn.Sequential(nn.Linear(proj_dim, proj_dim//4), nn.ReLU(),
                                     nn.Linear(proj_dim//4, proj_dim), nn.Sigmoid())
        self.gate_s = nn.Sequential(nn.Linear(proj_dim, proj_dim//4), nn.ReLU(),
                                     nn.Linear(proj_dim//4, proj_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(proj_dim*2, proj_dim),
                                     nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Dropout(dropout))

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
            "swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        swin_dim = self.swin_branch.num_features
        with torch.no_grad():
            dummy   = torch.zeros(1, 3, 256, 256)
            eff_dim = self.eff_branch(dummy).flatten(1).size(1)
        self.cross_attn_fs = CrossAttentionFS(eff_dim=eff_dim, swin_dim=swin_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):
        x224 = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        eff_feat  = self.eff_branch(x).flatten(1)
        swin_feat = self.swin_branch(x224)
        return self.classifier(self.cross_attn_fs(eff_feat, swin_feat))


# ─────────────────────────────────────────────────────────────
# CONVERT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OLD_PATH = "cn_detect_effb4_swin_best.pth"
    NEW_PATH = "cn_detect_webapp.pth"

    print("Loading old checkpoint...")
    device     = torch.device("cpu")
    old_state  = torch.load(OLD_PATH, map_location=device)

    print("Building model...")
    model = CNDetectEffB4Swin(num_classes=4)

    # Handle both old format (state_dict) and new format (dict with key)
    if isinstance(old_state, dict) and "model_state_dict" in old_state:
        model.load_state_dict(old_state["model_state_dict"])
        print("Loaded from complete checkpoint")
    else:
        model.load_state_dict(old_state)
        print("Loaded from state_dict")

    model.eval()

    # Build complete checkpoint
    complete = {
        "model_state_dict" : model.state_dict(),
        "model_config"     : {
            "num_classes"  : 4,
            "img_size"     : 256,
            "eff_dim"      : 1792,
            "swin_dim"     : 768,
            "proj_dim"     : 512,
            "num_heads"    : 8,
        },
        "classes"          : ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"],
        "val_accuracy"     : 95.80,
        "test_accuracy"    : 95.15,
        "normalize_mean"   : [0.485, 0.456, 0.406],
        "normalize_std"    : [0.229, 0.224, 0.225],
        "model_name"       : "EfficientNet-B4 + Swin-Tiny + CAFS",
        "architecture"     : "CN-Detect Hybrid CNN-Transformer",
        "description"      : "4-class chest X-ray classifier: COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS",
    }

    torch.save(complete, NEW_PATH)
    print(f"\n✅ Complete checkpoint saved → {NEW_PATH}")
    print(f"   Classes    : {complete['classes']}")
    print(f"   Test Acc   : {complete['test_accuracy']}%")
    print(f"   Val Acc    : {complete['val_accuracy']}%")
    print(f"   Model      : {complete['model_name']}")
    print(f"\n📌 This file is ready for web app deployment!")