"""Exact research architecture classes extracted from the historical module.
Legacy training/cache helpers are intentionally absent; ownership review remains pending.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception as exc:
    raise RuntimeError("mamba_ssm is required; fallback is forbidden") from exc

class MambaBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.m = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.m(self.norm(x))

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else torch.tensor([p])
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.view(x.size(0), -1)

class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if (not self.training) or (torch.rand(1).item() > self.p):
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + 1e-6).sqrt()
        x_normed = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return x_normed * sig_mix + mu_mix

class MultiScaleTokenMambaHead(nn.Module):
    def __init__(self, in_channels_list, d_model: int = 192, n_mamba: int = 1,
                 grid: int = 14, dropout: float = 0.2, num_classes: int = 3,
                 use_gem: bool = True):
        super().__init__()
        self.grid = grid
        self.use_gem = use_gem
        self.proj = nn.ModuleList([nn.Conv2d(c, d_model, kernel_size=1, bias=False) for c in in_channels_list])
        self.mamba = nn.Sequential(*[MambaBlock(d_model) for _ in range(n_mamba)])
        self.gem = GeM(p=3.0, trainable=True) if use_gem else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(len(in_channels_list) * d_model, num_classes)

    def _to_tokens(self, f):
        f = F.adaptive_avg_pool2d(f, (self.grid, self.grid))
        B, C, H, W = f.shape
        x = f.flatten(2).transpose(1, 2)
        return x, (B, C, H, W)

    def forward(self, feats):
        pooled = []
        for i, f in enumerate(feats):
            f = self.proj[i](f)
            tok, shape = self._to_tokens(f)
            tok = self.mamba(tok)
            B, C, H, W = shape
            f2 = tok.transpose(1, 2).reshape(B, C, H, W)
            if self.use_gem:
                pooled.append(self.gem(f2))
            else:
                pooled.append(F.adaptive_avg_pool2d(f2, 1).view(B, -1))
        x = torch.cat(pooled, dim=1)
        x = self.drop(x)
        return self.fc(x)

class OrchidEffLite2_MSMambaDG(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 taps=(3, 4), d_model: int = 192, n_mamba: int = 1,
                 mixstyle_p: float = 0.5, mixstyle_alpha: float = 0.1,
                 dropout: float = 0.2, use_gem: bool = True):
        super().__init__()
        preferred = ["tf_efficientnet_lite2.in1k", "tf_efficientnet_lite2"]
        avail = set(timm.list_models())
        name = next((n for n in preferred if n in avail), None)
        if name is None:
            cand = [m for m in timm.list_models() if ("efficientnet" in m and "lite2" in m and m.startswith("tf_"))]
            if not cand:
                raise RuntimeError("timm에서 EfficientNet-Lite2(tf_efficientnet_lite2) 계열 모델명을 찾지 못했습니다.")
            name = sorted(cand)[0]
        self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=taps)
        chs = self.backbone.feature_info.channels()
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.head = MultiScaleTokenMambaHead(
            in_channels_list=chs,
            d_model=d_model,
            n_mamba=n_mamba,
            grid=14,
            dropout=dropout,
            num_classes=num_classes,
            use_gem=use_gem,
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = [self.mixstyle(f) for f in feats]
        return self.head(feats)
