class BertConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        inner_size=3072,
        dropout_prob=0.1,
        max_seq_length=512,
        n_segments=2,
        pad_token_id=0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.inner_size = inner_size
        self.dropout_prob = dropout_prob
        self.max_seq_length = max_seq_length
        self.n_segments = n_segments
        self.pad_token_id = pad_token_id
