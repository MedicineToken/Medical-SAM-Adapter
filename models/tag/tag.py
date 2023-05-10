import math
import torch.nn.init as init
from timm.models.registry import register_model
from timm.models.layers import DropPath

from .tag_layers import *


class PatchEmbed(nn.Module):
    def __init__(self, stride, has_mask=False, in_ch=0, out_ch=0):
        super(PatchEmbed, self).__init__()
        self.to_token = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch)
        self.proj = nn.Linear(in_ch, out_ch, bias=False)
        self.has_mask = has_mask

    def process_mask(self, x, mask, H, W):
        if mask is None and self.has_mask:
            mask = x.new_zeros((1, 1, H, W))
        if mask is not None:
            H_mask, W_mask = mask.shape[-2:]
            if H_mask != H or W_mask != W:
                mask = F.interpolate(mask, (H, W), mode='nearest')
        return mask

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, 1, H, W] if exists, else None
        Returns:
            out: [B, out_H * out_W, out_C]
            H, W: output height & width
            mask: [B, 1, out_H, out_W] if exists, else None
        """
        out = self.to_token(x)
        B, C, H, W = out.shape
        mask = self.process_mask(out, mask, H, W)
        out = rearrange(out, "b c h w -> b (h w) c").contiguous()
        out = self.proj(out)
        return out, H, W, mask


class Encoder(nn.Module):
    def __init__(self, dim, num_parts=64, num_enc_heads=1, drop_path=0.1, act=nn.GELU, has_ffn=True):
        super(Encoder, self).__init__()
        self.num_heads = num_enc_heads
        self.enc_attn = AnyAttention(dim, num_enc_heads)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path else nn.Identity()
        self.reason = SimpleReasoning(num_parts, dim)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=act) if has_ffn else None

    def forward(self, feats, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            feats: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            qpos: [B, N, 1, C]
            kpos: [B, patch_num * patch_size, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        Returns:
            parts: [B, N, C]
        """
        attn_out = self.enc_attn(q=parts, k=feats, v=feats, qpos=qpos, kpos=kpos, mask=mask)
        parts = parts + self.drop_path(attn_out)
        parts = self.reason(parts)
        if self.enc_ffn is not None:
            parts = parts + self.drop_path(self.enc_ffn(parts))
        return parts


class Decoder(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=7, ffn_exp=3, act=nn.GELU, drop_path=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.attn1 = AnyAttention(dim, num_heads)
        self.attn2 = AnyAttention(dim, num_heads)
        self.rel_pos = FullRelPos(patch_size, patch_size, dim // num_heads)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act, norm_layer=Norm)
        self.ffn2 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act, norm_layer=Norm)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, parts=None, qpos=None, kpos=None, mask=None, P=0):
        """
        Args:
            x: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
            P: patch_num
        Returns:
            feat: [B, patch_num, patch_size, C]
        """
        dec_mask = None if mask is None else rearrange(mask.squeeze(1), "b h w -> b (h w) 1 1")
        out = self.attn1(q=x, k=parts, v=parts, qpos=qpos, kpos=kpos, mask=dec_mask)
        out = x + self.drop_path(out)
        out = out + self.drop_path(self.ffn1(out))

        # out = rearrange(out, "b (p k) c -> (b p) k c", p=P)
        # local_out = self.attn2(q=out, k=out, v=out, mask=mask, rel_pos=self.rel_pos)
        # out = out + self.drop_path(local_out)
        # out = out + self.drop_path(self.ffn2(out))
        return rearrange(out, "b (p k) c -> b p k c", p=P)


class TAGBlock(nn.Module):
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, patch_size=7, num_heads=1, num_enc_heads=1, num_parts=0):
        super(TAGBlock, self).__init__()
        # self.encoder = Encoder(dim, num_parts=num_parts, num_enc_heads=num_enc_heads, drop_path=drop_path)
        self.decoder = Decoder(dim, num_heads=num_heads, patch_size=patch_size, ffn_exp=ffn_exp, drop_path=drop_path)

    def forward(self, x, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            x: [B, patch_num, patch_size, C]
            parts: [B, N, C]
            part_qpos: [B, N, 1, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        Returns:
            feats: [B, patch_num, patch_size, C]
            parts: [B, N, C]
            part_qpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        """
        P = x.shape[1]
        x = rearrange(x, "b p k c -> b (p k) c")
        feats = self.decoder(x, parts=parts, qpos=qpos, kpos=kpos, mask=mask, P=P)
        return feats, parts, qpos, mask


class Stage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, patch_size=7, num_heads=1, num_enc_heads=1, stride=1, num_parts=0,
                 last_np=0, last_enc=False, drop_path=0.1, has_mask=None, ffn_exp=3):
        super(Stage, self).__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path for _ in range(num_blocks)]
        self.patch_size = patch_size
        self.rpn_qpos = nn.Parameter(torch.Tensor(1, num_parts, 1, out_ch // num_heads))
        self.rpn_kpos = nn.Parameter(torch.Tensor(1, num_parts, 1, out_ch // num_heads))

        self.proj_p = PatchEmbed(stride, has_mask = has_mask, in_ch=in_ch, out_ch=out_ch)
        self.proj_x = PatchEmbed(stride, has_mask = has_mask, in_ch=in_ch, out_ch=out_ch)
        # self.proj_token = nn.Sequential(
        #     nn.Conv1d(last_np, num_parts, 1, bias=False) if last_np != num_parts else nn.Identity(),
        #     nn.Linear(in_ch, out_ch),
        #     Norm(out_ch)
        # )
        self.proj_token = None
        self.proj_norm = Norm(out_ch)
        blocks = [
            TAGBlock(out_ch,
                     patch_size=patch_size,
                     num_heads=num_heads,
                     num_enc_heads=num_enc_heads,
                     num_parts=num_parts,
                     ffn_exp=ffn_exp,
                     drop_path=drop_path[i])
            for i in range(num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.last_enc = Encoder(dim=out_ch,
                                num_enc_heads=num_enc_heads,
                                num_parts=num_parts,
                                drop_path=drop_path[-1],
                                has_ffn=False) if last_enc else None
        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.rpn_qpos, a=math.sqrt(5))
        trunc_normal_(self.rpn_qpos, std=.02)
        init.kaiming_uniform_(self.rpn_kpos, a=math.sqrt(5))
        trunc_normal_(self.rpn_kpos, std=.02)

    def to_patch(self, x, patch_size, H, W, mask=None):
        x = rearrange(x, "b (h w) c -> b h w c", h=H)
        pad_l = pad_t = 0
        pad_r = int(math.ceil(W / patch_size)) * patch_size - W
        pad_b = int(math.ceil(H / patch_size)) * patch_size - H
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        if mask is not None:
            mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=1)
        x = rearrange(x, "b (sh kh) (sw kw) c -> b (sh sw) (kh kw) c", kh=patch_size, kw=patch_size)
        if mask is not None:
            mask = rearrange(mask, "b c (sh kh) (sw kw) -> b c (kh kw) (sh sw)", kh=patch_size, kw=patch_size)
        return x, mask, H + pad_b, W + pad_r
    
    def to_part(self, x, mask=None):
        x, H, W, mask = self.proj_p(x, mask=mask)
        x = self.proj_norm(x)
        if self.proj_token is not None:
            parts = self.proj_token(parts)
        ori_H, ori_W = H, W
        x, mask, H, W = self.to_patch(x, self.patch_size, H, W, mask)
        P = x.shape[1]
        x = rearrange(x, "b p k c -> b (p k) c")

        return x

    def forward(self, x, p, mask=None):
        """
        Args:
            x: [B, C, H, W]
            parts: [B, N, C]
            mask: [B, 1, H, W] if exists, else None
        Returns:
            x: [B, out_C, out_H, out_W]
            parts: [B, out_N, out_C]
            mask: [B, 1, out_H, out_W] if exists else None
        """
        parts = self.to_part(p, mask = mask)
        x, H, W, mask = self.proj_x(x, mask=mask)
        x = self.proj_norm(x)
        if self.proj_token is not None:
            parts = self.proj_token(parts)

        rpn_qpos, rpn_kpos = self.rpn_qpos, self.rpn_kpos
        rpn_qpos = rpn_qpos.expand(x.shape[0], -1, -1, -1)
        rpn_kpos = rpn_kpos.expand(x.shape[0], -1, -1, -1)

        ori_H, ori_W = H, W
        x, mask, H, W = self.to_patch(x, self.patch_size, H, W, mask)
        for blk in self.blocks:
            # x: [B, K, P, C]
            x, parts, rpn_qpos, mask = blk(x,
                                           parts=parts,
                                           qpos=rpn_qpos,
                                           kpos=rpn_kpos,
                                           mask=mask)

        dec_mask = None if mask is None else rearrange(mask.squeeze(1), "b h w -> b 1 1 (h w)")
        if self.last_enc is not None:
            x = rearrange(x, "b p k c -> b (p k) c")
            rpn_out = self.last_enc(x, parts=parts, qpos=rpn_qpos, mask=dec_mask)
            return rpn_out
        else:
            x = rearrange(x, "b (sh sw) (kh kw) c -> b c (sh kh) (sw kw)", kh=self.patch_size, sh=H // self.patch_size)
            x = x[:, :, :ori_H, :ori_W]
            return x


class TAG(nn.Module):
    def __init__(self,
                 in_chans=3,
                 inplanes=64,
                 num_layers=(3, 4, 6, 3),
                 num_chs=(256, 512, 1024, 2048),
                 num_strides=(1, 2, 2, 2),
                 num_classes=1000,
                 num_heads=(1, 1, 1, 1),
                 num_parts=(1, 1, 1, 1),
                 patch_sizes=(1, 1, 1, 1),
                 drop_path=0.1,
                 num_enc_heads=(1, 1, 1, 1),
                 act=nn.GELU,
                 ffn_exp=3,
                 no_pos_wd=False,
                 has_last_encoder=False,
                 pretrained=False,
                 **ret_args):
        super(TAG, self).__init__()
        self.depth = len(num_layers)
        self.no_pos_wd = no_pos_wd

        self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.act = act()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rpn_tokens = nn.Parameter(torch.Tensor(1, num_parts[0], inplanes))

        drop_path_ratios = torch.linspace(0, drop_path, sum(num_layers))
        last_chs = [inplanes, *num_chs[:-1]]
        last_nps = [num_parts[0], *num_parts[:-1]]

        for i, n_l in enumerate(num_layers):
            stage_ratios = [drop_path_ratios[sum(num_layers[:i]) + did] for did in range(n_l)]
            setattr(self,
                    "layer_{}".format(i),
                    Stage(last_chs[i],
                          num_chs[i],
                          n_l,
                          stride=num_strides[i],
                          num_heads=num_heads[i],
                          num_enc_heads=num_enc_heads[i],
                          patch_size=patch_sizes[i],
                          drop_path=stage_ratios,
                          ffn_exp=ffn_exp,
                          num_parts=num_parts[i],
                          last_np=last_nps[i],
                          last_enc=has_last_encoder and i == len(num_layers) - 1)
                    )

        if has_last_encoder:
            self.last_fc = nn.Linear(num_chs[-1], num_classes)
        else:
            self.last_linear = nn.Conv2d(num_chs[-1], num_chs[-1], kernel_size=1, bias=False)
            self.last_norm = nn.BatchNorm2d(num_chs[-1])
            self.pool2 = nn.AdaptiveAvgPool2d(1)
            self.last_fc = nn.Linear(num_chs[-1], num_classes)

        self.has_last_encoder = has_last_encoder
        self._init_weights(pretrained=pretrained)

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_pattern = ['rel_pos'] if self.no_pos_wd else []
        no_wd_layers = set()
        for name, param in self.named_parameters():
            for skip_name in skip_pattern:
                if skip_name in name:
                    no_wd_layers.add(name)
        return no_wd_layers

    def _init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location=torch.device("cpu"))
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=True)
            return

        init.kaiming_uniform_(self.rpn_tokens, a=math.sqrt(5))
        trunc_normal_(self.rpn_tokens, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not torch.sum(m.weight.data == 0).item() == m.num_features:  # zero gamma
                    m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.pool1(out)

        B, _, H, W = out.shape
        rpn_tokens, mask = self.rpn_tokens.expand(x.shape[0], -1, -1), None
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))
            out, rpn_tokens, mask = layer(out, rpn_tokens, mask=mask)

        if self.has_last_encoder:
            out = self.act(out)
            out = out.mean(1)
        else:
            out = self.last_linear(out)
            out = self.last_norm(out)
            out = self.act(out)
            out = self.pool2(out)
            out = out.squeeze()
        out = self.last_fc(out).squeeze()
        return out.view(out.size(0), -1)


@register_model
def TAG_mobile(pretrained=False, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(48, 96, 192, 384), patch_sizes=[8, 7, 7, 7], num_heads=[1, 2, 4, 8],
                     num_enc_heads=[1, 2, 4, 8], num_parts=[16, 16, 16, 32], num_layers=[1, 1, 1, 1], ffn_exp=3,
                     has_last_encoder=True, drop_path=0., **cfg)
    return TAG(pretrained=pretrained, **model_cfg)


@register_model
def TAG_tiny(pretrained=False, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(64, 128, 256, 512), patch_sizes=[8, 7, 7, 7], num_heads=[1, 2, 4, 8],
                     num_enc_heads=[1, 2, 4, 8], num_parts=[32, 32, 32, 32], num_layers=[1, 1, 2, 1], ffn_exp=3,
                     has_last_encoder=True, drop_path=0.1, **cfg)
    return TAG(pretrained=pretrained, **model_cfg)


@register_model
def TAG_small(pretrained=False, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(96, 192, 384, 768), patch_sizes=[8, 7, 7, 7], num_heads=[3, 6, 12, 24],
                     num_enc_heads=[1, 3, 6, 12], num_parts=[64, 64, 64, 64], num_layers=[1, 1, 3, 1], ffn_exp=3,
                     has_last_encoder=True, drop_path=0.1, **cfg)
    return TAG(pretrained=pretrained, **model_cfg)


@register_model
def TAG_medium(pretrained=False, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(96, 192, 384, 768), patch_sizes=[8, 7, 7, 7], num_heads=[3, 6, 12, 24],
                     num_enc_heads=[1, 3, 6, 12], num_parts=[64, 64, 64, 128], num_layers=[1, 1, 8, 1], ffn_exp=3,
                     has_last_encoder=False, drop_path=0.2, **cfg)
    return TAG(pretrained=pretrained, **model_cfg)


@register_model
def TAG_base(pretrained=False, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(128, 256, 512, 1024), patch_sizes=[8, 7, 7, 7], num_heads=[4, 8, 16, 32],
                     num_enc_heads=[1, 4, 8, 16], num_parts=[64, 64, 128, 128], num_layers=[1, 1, 8, 1], ffn_exp=3,
                     has_last_encoder=False, drop_path=0.3, **cfg)
    return TAG(pretrained=pretrained, **model_cfg)
