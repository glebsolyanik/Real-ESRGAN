from torch import nn
from torch.nn import functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RRDBNetX8(RRDBNet):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=4,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )
        self.scale = 8
        self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))

        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )

        return self.conv_last(self.lrelu(self.conv_hr(feat)))