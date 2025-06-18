'''
Here is a T2T-ViT model (for double images input).
'''
import torch
import torch.nn as nn
from timm.layers import DropPath


class DualT2T_Module(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        in_chans_main=2,
        in_chans_second=38,
        embed_dim=768,
        main_s0_channels=512,
        k_channels_per_pair_enhanced=4,
        second_s0_channels=128,
        fused_s1_channels=512,
        fused_s2_channels=128,
    ):
        super().__init__()
        self.soft_split0_main = nn.Conv2d(
            in_channels=in_chans_main,
            out_channels=main_s0_channels,
            kernel_size=7,
            stride=4,
            padding=3,
        )

        num_pairs = in_chans_second // 2

        self.pair_enhancer_conv = nn.Conv2d(
            in_channels=2,
            out_channels=k_channels_per_pair_enhanced,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.final_second_conv = nn.Conv2d(
            in_channels=num_pairs * k_channels_per_pair_enhanced,
            out_channels=second_s0_channels,
            kernel_size=7,
            stride=4,
            padding=3,
        )
        self.fusion_conv = nn.Conv2d(
            in_channels=main_s0_channels + second_s0_channels,
            out_channels=fused_s1_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.transformer1 = TransformerBlock(
            embed_dim=fused_s1_channels, num_heads=1, mlp_ratio=1.0
        )

        self.soft_split1 = nn.Conv2d(
            in_channels=fused_s1_channels,
            out_channels=fused_s2_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.transformer2 = TransformerBlock(
            embed_dim=fused_s2_channels, num_heads=1, mlp_ratio=1.0
        )

        self.soft_split2 = nn.Conv2d(
            in_channels=fused_s2_channels,
            out_channels=embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.num_patches = (img_size[0] // 16) * (img_size[1] // 16)

    def forward(self, x_main, x_second):
        B, _, H_img, W_img = x_main.shape
        num_pairs = (
            self.final_second_conv.in_channels // self.pair_enhancer_conv.out_channels
        )
        x1 = self.soft_split0_main(x_main)
        x_second_paired = x_second.contiguous().view(B, num_pairs, 2, H_img, W_img)
        x_second_for_enhancer = (
            x_second_paired.permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(-1, 2, H_img, W_img)
        )
        enhanced_pairs = self.pair_enhancer_conv(x_second_for_enhancer)
        k_enh = self.pair_enhancer_conv.out_channels
        x_combined_enhanced = enhanced_pairs.contiguous().view(
            B, num_pairs * k_enh, H_img, W_img
        )
        x2 = self.final_second_conv(x_combined_enhanced)
        x_concat = torch.cat((x1, x2), dim=1)
        x_fused = self.fusion_conv(x_concat)
        b, c, h, w = x_fused.shape
        x = x_fused.flatten(2).transpose(1, 2)
        x = self.transformer1(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.soft_split1(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer2(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.soft_split2(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class DualT2T_VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        in_chans_main=2,
        in_chans_second=38,
        main_s0_channels=512,
        k_channels_per_pair_enhanced=4,
        second_s0_channels=128,
        fused_s1_channels=512,
        fused_s2_channels=128,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0,
        droppath=0.01,
    ):
        super().__init__()

        self.patch_embed = DualT2T_Module(
            img_size=img_size,
            in_chans_main=in_chans_main,
            in_chans_second=in_chans_second,
            embed_dim=embed_dim,
            main_s0_channels=main_s0_channels,
            k_channels_per_pair_enhanced=k_channels_per_pair_enhanced,
            second_s0_channels=second_s0_channels,
            fused_s1_channels=fused_s1_channels,
            fused_s2_channels=fused_s2_channels,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        drop_path_rates = torch.linspace(0, droppath, depth).tolist()
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_main, x_second):
        B = x_main.shape[0]
        x = self.patch_embed(x_main, x_second)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class DualT2T_ViTRegressor(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        in_chans_main=2,
        in_chans_second=38,
        main_s0_channels=512,
        k_channels_per_pair_enhanced=4,
        second_s0_channels=128,
        fused_s1_channels=512,
        fused_s2_channels=128,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0,
        droppath=0.01,
        num_output_channels=3,
    ):
        super().__init__()
        self.vit = DualT2T_VisionTransformer(
            img_size=img_size,
            in_chans_main=in_chans_main,
            in_chans_second=in_chans_second,
            main_s0_channels=main_s0_channels,
            k_channels_per_pair_enhanced=k_channels_per_pair_enhanced,
            second_s0_channels=second_s0_channels,
            fused_s1_channels=fused_s1_channels,
            fused_s2_channels=fused_s2_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            droppath=droppath,
        )

        self.head = nn.Linear(embed_dim, num_output_channels)
        nn.init.zeros_(self.head.bias)
        self.output_dim = num_output_channels

    def forward(self, x_main, x_second):
        x = self.vit(x_main, x_second)
        out = self.head(x)
        return out
