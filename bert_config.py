class BertConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        layers=12,
        attention_heads=12,
        intermediate_size=3072,
        sequence_length=512,
        classes=2,
        segments=2,
        pad_token_id=0,
        dropout=0.1,
        norm_eps=1e-12,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.attention_heads = attention_heads
        self.intermediate_size = intermediate_size
        self.sequence_length = sequence_length
        self.classes = classes
        self.segments = segments
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.norm_eps = norm_eps
