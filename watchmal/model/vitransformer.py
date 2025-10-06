'''
    Here is a traditional ViT model.
'''
import torch
import torch.nn as nn
from timm.layers import DropPath


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        patch_size=16,
        in_chans=2,
        embed_dim=768,
        use_conv_stem=False,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        )
        self.img_size = img_size
        self.use_conv_stem = use_conv_stem
        self.patch_size = patch_size
        if use_conv_stem:
            self.grid_size = (img_size[0] // 8, img_size[1] // 8)
        else:
            self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        if self.use_conv_stem:
            x = self.stem(x)
        else:
            x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTRegressor(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        patch_size=16,
        in_chans=2,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        droppath=0.01,
        num_output_channels=3,
        use_conv_stem=False,
    ):
        super().__init__()

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_conv_stem=use_conv_stem,
        )

        self.head = nn.Linear(embed_dim, num_output_channels)
        nn.init.zeros_(self.head.bias)
        self.output_dim = num_output_channels

    def forward(self, x):
        x = self.vit(x)  # [B, embed_dim]
        out = self.head(x)  # [B, num_output_channels]

        return out


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


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        patch_size=16,
        in_chans=2,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        droppath=0.01,
        use_conv_stem=False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_conv_stem=use_conv_stem,
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

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
