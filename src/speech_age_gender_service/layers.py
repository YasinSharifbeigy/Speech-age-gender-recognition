import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttentionMultiHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        if mask is not None:
            raise ValueError("Attention masks are not supported yet")

        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.shape[-1])
        attention_weights = self.softmax(attention_weights)
        return torch.matmul(attention_weights, value), attention_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(F.relu(self.dropout(self.layer_1(x))))


class AddAndNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float | None = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)

        return self.layer_norm(x + residual)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, dropout: float | None = 0.1) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.keys_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.values_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.ff_layer_after_concat = nn.Linear(self.num_heads * self.head_dim, input_dim, bias=False)
        self.attention = ScaledDotProductAttentionMultiHead()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, len_query, len_keys, len_values = queries.size(0), queries.size(1), keys.size(1), values.size(1)
        queries = self.query_w(queries).view(batch_size, len_query, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.keys_w(keys).view(batch_size, len_keys, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.values_w(values).view(batch_size, len_values, self.num_heads, self.head_dim).transpose(1, 2)
        values, _attention_weights = self.attention(queries, keys, values, mask=mask)
        out = values.transpose(1, 2).contiguous().view(batch_size, len_values, self.num_heads * self.head_dim)
        return self.ff_layer_after_concat(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._build_pe(), persistent=True)

    def _build_pe(self) -> torch.Tensor:
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe.permute(1, 0, 2)

    def reset_parameters(self) -> None:
        self.pe.copy_(self._build_pe().to(device=self.pe.device, dtype=self.pe.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, dropout: float | None = 0.1, positional_encoding: bool = True):
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None
        self.self_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout or 0.0)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor, mask=None) -> torch.Tensor:
        if self.positional_encoding is not None:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        x = self.self_attention(queries=query, keys=key, values=value, mask=mask)
        x = self.add_norm_after_attention(x, query)
        return self.add_norm_after_ff(self.feed_forward(x), x)


class StatPoolLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=self.dim), x.std(dim=self.dim, correction=0)], dim=-1)


class AGenderClassificationHead(nn.Module):
    def __init__(self, input_size: int = 256, output_size: int = 4) -> None:
        super().__init__()
        self.fc_agender = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_agender = self.fc_agender(x)
        return {"gen": x_agender[:, 0:-1], "age": x_agender[:, -1]}


class AGenderClassificationHeadV2(nn.Module):
    def __init__(self, input_size: int = 256, gender_classes: int = 2) -> None:
        super().__init__()
        self.fc_gender = nn.Linear(input_size, gender_classes)
        self.fc_age = nn.Linear(input_size, 1)

    def forward(self, x_gender: torch.Tensor, x_age: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"gen": self.fc_gender(x_gender), "age": self.fc_age(x_age).squeeze(-1)}


class MaskAGenderClassificationHead(nn.Module):
    def __init__(self, input_size: int = 256, output_size: int = 3, mask_classes: int = 6) -> None:
        super().__init__()
        self.fc_agender = nn.Linear(input_size, output_size)
        self.fc_mask = nn.Linear(input_size, mask_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_agender = self.fc_agender(x)
        return {"gen": x_agender[:, 0:-1], "age": x_agender[:, -1], "mask": self.fc_mask(x)}


class Permute(nn.Module):
    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)
