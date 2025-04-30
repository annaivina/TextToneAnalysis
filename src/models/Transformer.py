import torch 


class TransformerEncoder(torch.nn.Module):
  def __init__(self, embed_dim, dense_dim, num_heads, dropout):
    super(TransformerEncoder, self).__init__()

    self.attnetion = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    self.norm1 = torch.nn.LayerNorm(embed_dim)
    self.norm2 = torch.nn.LayerNorm(embed_dim)
    self.linear = torch.nn.Sequential(torch.nn.Linear(embed_dim, dense_dim),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(dense_dim, embed_dim))
    self.dropout = torch.nn.Dropout(dropout)


  def forward(self, input, mask=None):
    #Transform input because it is in the shape of (batch,sec_len,embed_dim)
    x = input.transpose(0, 1)  # (seq_len, batch, embed_dim)
    
    # mask: (batch, seq_len) → key_padding_mask
    key_padding_mask = None
    if mask is not None:
        key_padding_mask = ~mask.bool()  # PyTorch expects True = mask out
    
    out_1,_ = self.attnetion(x, x, x, attn_mask=key_padding_mask)
    x = self.norm1(out_1 + x)#add the attention + input (aka embedding)
    out = self.linear(x)
    out = self.dropout(out)
    out = self.norm2(out + x)

    
    return out.transpose(0, 1)  # (batch, seq_len, embed_dim)