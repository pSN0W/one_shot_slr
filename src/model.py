import einops
import math
import torch
import torch.nn as nn
from torch import Tensor


class KeypointSiameseNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        max_sequence_length: int,
        encoders: list[int],
        fcc: list[int],
        num_heads: int,
        expansion: float,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            feature_dim=feature_dim,
            max_sequence_length=max_sequence_length,
            encoders=encoders,
            fcc=fcc,
            num_heads=num_heads,
            expansion=expansion,
            dropout_p=dropout_p,
        )

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        if y is None:
            return self.encoder(x)
        encoded_x = self.encoder(x)
        encoded_y = self.encoder(y)
        return encoded_x, encoded_y

class FFD(nn.Sequential):
    def __init__(self, x,y, p):
        super().__init__(
            nn.LayerNorm(x),
            nn.Linear(x,y),
            nn.GELU(),
            nn.Dropout(p)
        )

class Encoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        max_sequence_length: int,
        encoders: list[int],
        fcc: list[int],
        num_heads: int,
        expansion: float,
        dropout_p: float,
    ) -> None:
        super().__init__()
        initial_encodig = encoders[0]

        # to converate the cordinates of x,y,z into another space
        self.cordinate_transformer = nn.Linear(3, 3)

        self.normalization = nn.LayerNorm(feature_dim)
        # Initial encoder to encoder the output
        self.cordinate_encoder = nn.Linear(feature_dim, initial_encodig)

        # positional encoder
        self.positinal_encoder = PositionalEncoder(
            d_model=initial_encodig, max_len=max_sequence_length, dropout=dropout_p
        )

        # transformer block
        self.transformer_encoder = TransformerEncoder(
            encoding_dimensions=encoders,
            num_heads=num_heads,
            expansion=expansion,
            dropout_p=dropout_p,
        )
        fcc = [encoders[-1]] + fcc

        self.lstm = nn.LSTM(initial_encodig, initial_encodig, 4, batch_first=True)
        # final linear block
        self.linear = nn.Sequential(
            *([FFD(x,y, dropout_p) for x,y in zip(fcc[:-1], fcc[1:])] + [nn.Linear(fcc[-1], fcc[-1])])
        )
        

    def forward(self, x: Tensor) -> Tensor:
        # Apply cordinate transformation on the cordinates of landmarks
        # x = einops.rearrange(x, "b s (num_key cord) -> b s num_key cord", cord=3)

        # x = self.cordinate_transformer(x)
        # x = einops.rearrange(x, "b s num_key cord -> b s (num_key cord)")

        # compress the cordinates into required form
        x = self.cordinate_encoder(self.normalization(x))

        # add positional encoding to each of the cordinates
        # x = self.positinal_encoder(x)

        # # pass it through the transformer encoder
        x = self.transformer_encoder(x)
        
        # x,_ = self.lstm(x)

        # take max of the featured for all the sequence
        x = einops.reduce(x, "b s feat -> b feat", "max")

        x = self.linear(x)

        # normalize the scores
        x = x / torch.linalg.norm(x, axis=-1, keepdim=True)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # scale the values
        out = x * math.sqrt(self.d_model)

        # add the positional encodings
        out = out + self.pe[: x.size(1)]

        # add dropouts
        out = self.dropout(out)

        return out


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        encoding_dimensions: list[int],
        num_heads: int,
        expansion: float,
        dropout_p: float,
    ):
        assert (
            len(encoding_dimensions) > 1
        ), "Encoding dimension must have length greater then 1"
        super().__init__(
            *[
                TransformerEncoderBlock(
                    in_dim=encoding_dimensions[i],
                    out_dim=encoding_dimensions[i + 1],
                    num_heads=num_heads,
                    expansion=expansion,
                    droput_p=dropout_p,
                )
                for i in range(len(encoding_dimensions) - 1)
            ]
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        expansion: float,
        droput_p: float,
    ):
        super().__init__(
            ResidualConnection(
                nn.LayerNorm(in_dim),
                WrappedMultiheadAttention(embed_dim=in_dim, num_heads=num_heads),
                nn.Dropout(droput_p),
            ),
            ResidualConnection(
                nn.LayerNorm(in_dim),
                AttentionFeedForwardBlock(
                    num_feat=in_dim, expansion=expansion, drop_p=droput_p
                ),
                nn.Dropout(droput_p),
            ),
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
        )


class ResidualConnection(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.Sequential(*fns)

    def forward(self, x, **kwargs):
        identity = x
        return identity + self.fns(x, **kwargs)


class WrappedMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        x, attn_weights = super().forward(x, x, x, need_weights=False)
        return x


class AttentionFeedForwardBlock(nn.Sequential):
    def __init__(self, num_feat: int, expansion: int, drop_p: float = 0):
        super().__init__(
            nn.Linear(num_feat, int(num_feat * expansion)),
            nn.GELU(),
            nn.Linear(int(num_feat * expansion), num_feat),
            nn.Dropout(p=drop_p),
        )
