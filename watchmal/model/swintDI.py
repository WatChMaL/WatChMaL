"""
Swin Transformer for Dual Image (SwinT DI) model.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_
from typing import Tuple
import numpy as np


def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / (window_size * window_size)))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PairMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, pairs):
        B, N, P, _ = pairs.shape
        x = pairs.reshape(B * N * P, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.reshape(B, N, P, -1)
        return x


class ResidualFFN(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class ConvStem(nn.Module):
    def __init__(
        self, in_chans=2, pre_stage_embed_dim=48, stem_channels=(32,), act_layer=nn.GELU
    ):
        super().__init__()
        layers = []
        c_in = in_chans
        c_out = stem_channels[0] if len(stem_channels) > 0 else pre_stage_embed_dim
        layers += [nn.Conv2d(c_in, c_out, 3, 2, 1, bias=True), act_layer()] 
        c_in = c_out
        layers += [
            nn.Conv2d(c_in, pre_stage_embed_dim, 3, 1, 1, bias=True),
            act_layer(),
        ]
        layers += [
            nn.Conv2d(pre_stage_embed_dim, pre_stage_embed_dim, 3, 1, 1, bias=True),
            act_layer(),
        ]
        layers += [
            nn.Conv2d(pre_stage_embed_dim, pre_stage_embed_dim, 1, 1, 0, bias=True)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        patch_size=1,
        in_chans=48,
        embed_dim=48,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.patches_resolution = patches_resolution
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=6,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W) 

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample
            else None
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class CrossWindowBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        window_size=4,
        num_heads=6,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        shift_size=0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.ws = window_size
        self.HW = input_resolution
        self.h = num_heads
        self.dh = dim // num_heads
        self.scale = (self.dh) ** -0.5
        self.shift_size = shift_size

        self.norm_q = nn.LayerNorm(dim)
        self.norm_km = nn.LayerNorm(dim)
        self.norm_kp = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(drop)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            H, W = self.HW
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.ws),
                slice(-self.ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.ws),
                slice(-self.ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.ws).view(
                -1, self.ws * self.ws
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    @staticmethod
    def _bnw(x, H, W, C, ws):
        x = x.view(-1, H, W, C)
        xw = window_partition(x, ws).view(-1, ws * ws, C)
        return xw

    @staticmethod
    def _rev(xw, H, W, C, ws):
        xw = xw.view(-1, ws, ws, C)
        x = window_reverse(xw, ws, H, W).view(-1, H * W, C)
        return x

    def forward(self, x_main, x_mpmt):
        B, N, C = x_main.shape
        H, W = self.HW
        assert N == H * W

        q = self.norm_q(x_main)
        km = self.norm_km(x_main)
        kp = self.norm_kp(x_mpmt)

        if self.shift_size > 0:

            def roll_2d(t):
                return torch.roll(
                    t.view(B, H, W, C),
                    shifts=(-self.shift_size, -self.shift_size),
                    dims=(1, 2),
                ).view(B, H * W, C)

            q, km, kp = roll_2d(q), roll_2d(km), roll_2d(kp)

        q_w = self._bnw(q, H, W, C, self.ws)
        km_w = self._bnw(km, H, W, C, self.ws)
        kp_w = self._bnw(kp, H, W, C, self.ws)
        kv_w = torch.cat([km_w, kp_w], dim=1).contiguous()

        def proj_heads(x, proj, h):
            x = proj(x)
            x = x.view(x.size(0), x.size(1), h, -1).permute(0, 2, 1, 3).contiguous()
            return x

        Q = proj_heads(q_w, self.q_proj, self.h)
        K = proj_heads(kv_w, self.k_proj, self.h)
        V = proj_heads(kv_w, self.v_proj, self.h)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        if self.attn_mask is not None:
            nW, Lq, _ = self.attn_mask.shape
            attn_mask2 = torch.cat([self.attn_mask, self.attn_mask], dim=-1)
            scores = scores.view(B, nW, self.h, Lq, 2 * Lq)
            scores = scores + attn_mask2.unsqueeze(0).unsqueeze(2)
            scores = scores.view(-1, self.h, Lq, 2 * Lq)
        attn = scores.softmax(dim=-1)
        out = attn @ V

        out = (
            out.permute(0, 2, 1, 3).contiguous().view(out.size(0), -1, self.h * self.dh)
        )
        y = self._rev(out, H, W, C, self.ws)

        if self.shift_size > 0:
            y = torch.roll(
                y.view(B, H, W, C),
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            ).view(B, N, C)

        x = x_main + self.drop(self.o_proj(y))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class SparseMpmtEncoder(nn.Module):
    def __init__(self, out_token_dim=48, pair_hidden=8, ffn_hidden=128, drop=0.0):
        super().__init__()
        self.pair_hidden = pair_hidden
        self.pair_mlp = PairMLP(hidden=pair_hidden)

        in_dim_after_pairs = 19 * pair_hidden + 2

        self.ffn1 = ResidualFFN(in_dim_after_pairs, ffn_hidden, drop=drop)
        self.ffn2 = ResidualFFN(in_dim_after_pairs, ffn_hidden, drop=drop)

        self.norm = nn.LayerNorm(in_dim_after_pairs)

        self.head = nn.Linear(in_dim_after_pairs, out_token_dim)

    def forward(self, x_sparse, xy_coords):
        x = x_sparse.permute(0, 2, 1).contiguous()
        B, N, _ = x.shape

        pairs = x.view(B, N, 19, 2)

        pair_embed = self.pair_mlp(pairs)

        pair_embed = pair_embed.flatten(2)

        xy_normalized = (xy_coords.float() / 191.0) * 2 - 1

        h = torch.cat([pair_embed, xy_normalized], dim=-1)
        h = self.ffn1(h)
        h = self.ffn2(h)
        h = self.norm(h)

        tokens = self.head(h)

        return tokens


class SwinTransformerDI(nn.Module):
    def __init__(
        self,
        mpmt_lut_path,
        img_size=(192, 192),
        in_chans_main=2,
        in_chans_mpmt=38,
        pre_stage_embed_dim=48,
        pre_stage_depth=8,
        pre_stage_num_heads=6,
        pre_stage_window_size=4,
        xattn_first_k_layers=2,
        xattn_num_heads=6,
        embed_dim=96,
        depths=[8, 8, 6, 2],
        num_heads=[6, 8, 12, 24],
        window_size=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        ape=True,
        patch_norm=True,
        pair_hidden=8,
        ffn_hidden=128,
        use_conv_stem=False,
    ):
        super().__init__()
        lut_numpy = np.load(mpmt_lut_path)["pmt_module_positions"]
        self.register_buffer("mpmt_lut", torch.from_numpy(lut_numpy).long())
        self.xattn_first_k_layers = int(xattn_first_k_layers)
        self.pre_stage_embed_dim = pre_stage_embed_dim
        stem_out_img = (img_size[0] // 2, img_size[1] // 2)
        self.use_conv_stem = use_conv_stem
        if self.use_conv_stem:
            self.stem_main = ConvStem(
                in_chans=in_chans_main, pre_stage_embed_dim=pre_stage_embed_dim
            )
            self.patch_embed_main = PatchEmbed(
                img_size=stem_out_img,
                patch_size=1,
                in_chans=pre_stage_embed_dim,
                embed_dim=pre_stage_embed_dim,
                norm_layer=norm_layer if patch_norm else None,
            )
        else:
            self.stem_main = nn.Identity()
            self.patch_embed_main = PatchEmbed(
                img_size=img_size,
                patch_size=2,
                in_chans=in_chans_main,
                embed_dim=pre_stage_embed_dim,
                norm_layer=norm_layer if patch_norm else None,
            )

        self.mpmt_encoder = SparseMpmtEncoder(
            out_token_dim=pre_stage_embed_dim,
            pair_hidden=pair_hidden,
            ffn_hidden=ffn_hidden,
            drop=drop_rate,
        )

        num_patches_pre = (
            self.patch_embed_main.patches_resolution[0]
            * self.patch_embed_main.patches_resolution[1]
        )
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches_pre, pre_stage_embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.pre_blocks_tail = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=pre_stage_embed_dim,
                    input_resolution=tuple(self.patch_embed_main.patches_resolution),
                    num_heads=pre_stage_num_heads,
                    window_size=pre_stage_window_size,
                    shift_size=0 if ((i) % 2 == 0) else pre_stage_window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.0,
                )
                for i in range(self.xattn_first_k_layers, pre_stage_depth)
            ]
        )

        self.xattn_blocks = nn.ModuleList(
            [
                CrossWindowBlock(
                    dim=pre_stage_embed_dim,
                    input_resolution=tuple(self.patch_embed_main.patches_resolution),
                    window_size=pre_stage_window_size,
                    num_heads=xattn_num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    shift_size=0 if (i % 2 == 0) else pre_stage_window_size // 2,
                )
                for i in range(self.xattn_first_k_layers)
            ]
        )

        self.token_fuse = nn.Sequential(
            nn.LayerNorm(pre_stage_embed_dim * 2),
            nn.Linear(pre_stage_embed_dim * 2, 2 * pre_stage_embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(2 * pre_stage_embed_dim, pre_stage_embed_dim),
        )
        self.pre_stage_merging = PatchMerging(
            input_resolution=tuple(self.patch_embed_main.patches_resolution),
            dim=pre_stage_embed_dim,
            norm_layer=norm_layer,
        )
        self.proj_to_stage1 = nn.Linear(2 * pre_stage_embed_dim, embed_dim, bias=False)

        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        current_dim = embed_dim
        current_resolution = (
            self.patch_embed_main.patches_resolution[0] // 2,
            self.patch_embed_main.patches_resolution[1] // 2,
        )
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=current_dim,
                input_resolution=current_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                current_resolution = (
                    current_resolution[0] // 2,
                    current_resolution[1] // 2,
                )
                current_dim = int(current_dim * 2)

        self.num_features = current_dim
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_main: torch.Tensor, x_mpmt38: torch.Tensor):
        B = x_main.shape[0]

        xm_img = self.stem_main(x_main)
        xm = self.patch_embed_main(xm_img)
        xy_coords = self.mpmt_lut.unsqueeze(0).expand(B, -1, -1)
        mpmt_tokens = self.mpmt_encoder(x_mpmt38, xy_coords)

        num_patches = (
            self.patch_embed_main.patches_resolution[0]
            * self.patch_embed_main.patches_resolution[1]
        )
        mm = torch.zeros(
            B,
            num_patches,
            self.pre_stage_embed_dim,
            device=mpmt_tokens.device,
            dtype=mpmt_tokens.dtype,
        )
        H_feat, W_feat = self.patch_embed_main.patches_resolution
        rows = xy_coords[:, :, 0].clamp(0, H_feat - 1)
        cols = xy_coords[:, :, 1].clamp(0, W_feat - 1)
        flat_indices = rows * W_feat + cols

        b_idx = torch.arange(B, device=x_main.device)[:, None].expand_as(flat_indices)

        mm[b_idx, flat_indices] = mpmt_tokens

        if self.absolute_pos_embed is not None:
            xm = xm + self.absolute_pos_embed
            B = x_main.size(0)
            num_patches = (
                self.patch_embed_main.patches_resolution[0]
                * self.patch_embed_main.patches_resolution[1]
            )
            occ = torch.zeros(
                B, num_patches, 1, device=mpmt_tokens.device, dtype=mpmt_tokens.dtype
            )
            b_idx = torch.arange(B, device=x_main.device)[:, None].expand_as(
                flat_indices
            ) 
            occ[b_idx, flat_indices] = 1
            mm = mm + self.absolute_pos_embed * occ
        xm = self.pos_drop(xm)
        mm = self.pos_drop(mm)

        for i in range(self.xattn_first_k_layers):
            xm = self.xattn_blocks[i](xm, mm)

        xcat = torch.cat([xm, mm], dim=-1)
        xm = self.token_fuse(xcat)

        for blk in self.pre_blocks_tail:
            xm = blk(xm)

        xm = self.pre_stage_merging(xm)
        xm = self.proj_to_stage1(xm)

        for layer in self.layers:
            xm = layer(xm)
        xm = self.norm(xm)
        xm = self.avgpool(xm.transpose(1, 2))
        xm = torch.flatten(xm, 1)
        return xm


class SwinRegressorDI(nn.Module):
    def __init__(self, mpmt_lut_path, num_output_channels=3, **kwargs):
        super().__init__()
        self.backbone = SwinTransformerDI(mpmt_lut_path, **kwargs)
        self.head = nn.Linear(self.backbone.num_features, num_output_channels)
        nn.init.zeros_(self.head.bias)
        self.output_dim = num_output_channels

    def forward(self, x_main, x_mpmt38):
        feats = self.backbone(x_main, x_mpmt38)
        return self.head(feats)
