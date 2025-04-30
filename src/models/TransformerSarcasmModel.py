import torch
from src.models.customEmbed import CustomEmbedding
from src.models.Transformer import TransformerEncoder


class SarcasmTransformer(torch.nn.Module):
  def __init__(self, voc_size, seq_length, embed_dim, dense_dim, num_heads, dropout, num_layers):
    super(SarcasmTransformer, self).__init__()

    self.embed = CustomEmbedding(voc_size, embed_dim, seq_length)
    self.encoder = torch.nn.ModuleList([TransformerEncoder(embed_dim, dense_dim, num_heads, dropout) for _ in range(num_layers)])
    self.linear = torch.nn.Sequential(torch.nn.Linear(embed_dim, dense_dim),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(dense_dim, 2))
    self.max_pool = torch.nn.AdaptlsiveMaxPool1d(1)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x, mask = None):
    x = self.embed(x)
    #Encoder layers
    for layer in self.encoder:
      x = layer(x, mask)

    #trandform to be able to use it in Linear layer
    x = x.transpose(1, 2)            # (batch, embed_dim, seq_len)
    x = self.max_pool(x)             # (batch, embed_dim, 1)
    x = x.squeeze(-1) 
    x  = self.dropout(x)
    x = self.linear(x)

    return x