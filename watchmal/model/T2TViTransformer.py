'''
Here is a T2T-ViT model (for single image input).
'''
import torch
import torch.nn as nn
from timm.layers import DropPath


class T2T_Module(nn.Module):
    def __init__(
        self, img_size=(192, 192), in_chans=2, embed_dim=768, token_dims=[64, 128]
    ):
        super().__init__()
        self.token_dims = token_dims
        self.soft_split0 = nn.Conv2d(
            in_chans, token_dims[0], kernel_size=7, stride=4, padding=3
        )
        self.transformer1 = TransformerBlock(
            embed_dim=token_dims[0], num_heads=1, mlp_ratio=1.0
        )
        h_s1, w_s1 = img_size[0] // 4, img_size[1] // 4
        self.soft_split1 = nn.Conv2d(
            token_dims[0], token_dims[1], kernel_size=3, stride=2, padding=1
        )
        self.transformer2 = TransformerBlock(
            embed_dim=token_dims[1], num_heads=1, mlp_ratio=1.0
        )
        h_s2, w_s2 = h_s1 // 2, w_s1 // 2
        self.soft_split2 = nn.Conv2d(
            token_dims[1], embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.num_patches = (h_s2 // 2) * (w_s2 // 2)

    def forward(self, x):
        x = self.soft_split0(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
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


class T2T_VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        in_chans=2,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        droppath=0.01,
        t2t_token_dims=[64, 128],
    ):
        super().__init__()

        self.patch_embed = T2T_Module(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dims=t2t_token_dims,
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
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class T2T_ViTRegressor(nn.Module):
    def __init__(
        self,
        img_size=(192, 192),
        in_chans=2,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        droppath=0.01,
        t2t_token_dims=[64, 128],
        num_output_channels=3,
    ):
        super().__init__()
        self.vit = T2T_VisionTransformer(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            droppath=droppath,
            t2t_token_dims=t2t_token_dims,
        )

        self.head = nn.Linear(embed_dim, num_output_channels)
        nn.init.zeros_(self.head.bias)
        self.output_dim = num_output_channels

    def forward(self, x):
        x = self.vit(x)
        out = self.head(x)
        return out
