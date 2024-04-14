import torch
from torch import nn


class BertEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        sequence_length,
        token_types=2,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(sequence_length, hidden_size)
        self.token_type_embeddings = nn.Embedding(token_types, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_ids, token_type_ids, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(token_ids.size(1)).expand((1, -1))
        embeddings = (
            self.word_embeddings(token_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings(position_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.head_dim = hidden_size // attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.attention_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, inputs, mask=None):
        q, k, v = inputs, inputs, inputs
        batch_size = q.size(0)

        q = self.transpose_for_scores(self.query(q))
        k = self.transpose_for_scores(self.key(k))
        v = self.transpose_for_scores(self.value(v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float)
        )

        if mask is not None:
            mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        )
        return context


class AttentionOutput(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, context):
        outputs = self.dense(context)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm(outputs + inputs)
        return outputs


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, attention_heads):
        super().__init__()
        self.self = SelfAttention(hidden_size, attention_heads)
        self.output = AttentionOutput(hidden_size)

    def forward(self, inputs, mask):
        context = self.self(inputs, mask)
        outputs = self.output(inputs, context)
        return outputs


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, inputs):
        outputs = self.dense(inputs)
        outputs = nn.functional.gelu(outputs)
        return outputs


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, intermediate_output, attention_output):
        outputs = self.dense(intermediate_output)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm(outputs + attention_output)
        return outputs


class BertEncoder(nn.Module):
    def __init__(
        self, attention_heads, hidden_size, intermediate_size, dropout_rate=0.1
    ):
        super(BertEncoder, self).__init__()
        self.attention = AttentionBlock(hidden_size, attention_heads)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size)

    def forward(self, inputs, mask):
        attention_output = self.attention(inputs, mask)
        intermediate_output = self.intermediate(attention_output)
        output = self.output(intermediate_output, attention_output)
        return output


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        first_token_tensor = inputs[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        max_seq_length,
        attention_heads,
        num_layers,
    ):
        super(Bert, self).__init__()
        self.embeddings = BertEmbedding(vocab_size, hidden_size, max_seq_length)
        self.encoder = nn.ModuleDict(
            [
                [
                    f"layer_{str(i)}",
                    BertEncoder(attention_heads, hidden_size, intermediate_size),
                ]
                for i in range(num_layers)
            ]
        )
        self.pooler = BertPooler(hidden_size)

    def forward(self, tokens, segments, attention_mask):
        encoded_tokens = self.embeddings(tokens, segments)
        for _, encoder_block in self.encoder.items():
            encoded_tokens = encoder_block(encoded_tokens, attention_mask)
        pooled_output = self.pooler(encoded_tokens)
        return encoded_tokens, pooled_output


class BertTransformHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = BertTransformHead(hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.dense.bias = self.output_bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.dense(hidden_states)
        return hidden_states


class CLS(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_classes=2):
        super().__init__()
        self.predictions = BertPredictionHead(hidden_size, vocab_size)
        self.seq_relationship = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertForPreTraining(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        max_seq_length,
        attention_heads,
        num_layers,
        num_classes,
    ):
        super().__init__()
        self.bert = Bert(
            vocab_size,
            hidden_size,
            intermediate_size,
            max_seq_length,
            attention_heads,
            num_layers,
        )
        self.cls = CLS(hidden_size, vocab_size, num_classes)

    def forward(self, tokens, segments, attention_mask):
        sequence_output, pooled_output = self.bert(tokens, segments, attention_mask)

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        return sequence_output, pooled_output, prediction_scores, seq_relationship_score
