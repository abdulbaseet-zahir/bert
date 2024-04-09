import math
import torch
from torch import nn
import math
import torch
from torch import nn
import torch.nn.functional as F


class BertEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_seq_length,
        n_segments=2,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.segment_embedding = nn.Embedding(n_segments, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)
        self.position_ids = torch.arange(max_seq_length).expand((1, -1))
        self.segment_ids = torch.zeros(self.position_ids.size(), dtype=torch.long)

    def forward(self, input_ids, segment_ids):
        embeddings = (
            self.token_embedding(input_ids)
            + self.segment_embedding(segment_ids)
            + self.position_embedding(self.position_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)

        query = self.transpose_for_scores(self.query_projection(queries))
        key = self.transpose_for_scores(self.key_projection(keys))
        value = self.transpose_for_scores(self.value_projection(values))

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float)
        )

        if mask is not None:
            mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, value)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return context


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout_rate=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tensor):
        tensor = F.gelu(self.fc1(tensor))
        tensor = self.dropout(tensor)
        tensor = self.fc2(tensor)
        return tensor


class BertBlock(nn.Module):
    def __init__(self, attention_heads, hidden_size, inner_size, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(attention_heads, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = PositionWiseFeedForward(hidden_size, inner_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        projected_output = self.projection(attention_output)
        norm1_output = self.norm1(x + self.dropout(projected_output))
        feed_forward_output = self.feed_forward(norm1_output)
        norm2_output = self.norm2(norm1_output + self.dropout(feed_forward_output))
        return norm2_output


class BertEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_size,
        max_seq_length,
        attention_heads,
        num_layers,
    ):
        super(BertEncoder, self).__init__()
        self.embedding = BertEmbedding(vocab_size, hidden_size, max_seq_length)
        self.encoder_blocks = nn.ModuleList(
            [
                BertBlock(attention_heads, hidden_size, inner_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens, segments, attention_mask):
        encoded_tokens = self.embedding(tokens, segments)
        for encoder_block in self.encoder_blocks:
            encoded_tokens = encoder_block(encoded_tokens, attention_mask)
        return encoded_tokens
