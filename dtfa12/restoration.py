"""Local IRT reconstruction recipe, not the unavailable authors' decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .models import IRTEncoder


class RestorationTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IRTEncoder()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())
        self.deep = block(128, 64)
        self.fuse = block(128, 64)
        self.up2 = block(64, 32)
        self.output = nn.Sequential(block(32, 16), nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, rgb):
        p3, p2 = self.encoder(rgb)
        x = F.interpolate(self.deep(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat((x, p2), dim=1))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.output(F.interpolate(x, size=rgb.shape[-2:], mode="bilinear", align_corners=False))


def reconstruction_loss(model, fog, clear, identity_weight):
    if identity_weight:
        restored, identity = model(torch.cat((fog, clear))).chunk(2)
        identity_loss = F.l1_loss(identity, clear)
    else:
        restored = model(fog)
        identity_loss = fog.new_zeros(())
    restoration_loss = F.l1_loss(restored, clear)
    return restoration_loss + identity_weight * identity_loss, restoration_loss, identity_loss


def psnr(prediction, target):
    """Mean per-image RGB PSNR on the resized/letterboxed [0,1] tensors."""
    mse = (prediction - target).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10 * mse.log10()
