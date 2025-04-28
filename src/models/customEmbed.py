import torch
import math 


class CustomEmbedding(torch.nn.Module):
  def __init__(self, vocab_size, embed_dim, seq_length):
    super(CustomEmbedding, self).__init__()

    self.embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
    # Register positional encoding as a buffer so it moves with .to(device)
    pe = positional_encoding(embed_dim, seq_length)
    self.register_buffer("pos_embed", pe.to(dtype=torch.float32))  # not a learnable param


  def forward(self, x):
    embeded = self.embed(x)
    position = self.pos_embed[:, :x.size(1), :].to(x.device)
    return embeded + position
    #shape is (1, seq_length, embed_dim)




def positional_encoding(model_size, seq_length):

    """
    Create sinusoidal positional encodings.

    Args:
        model_size (int): the embedding dimension
        seq_length (int): maximum sequence length

    Returns:
        torch.Tensor: positional encodings with shape (1, seq_length, model_size)
    """

    pe = torch.zeros(seq_length, model_size)

    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)  
    div_term = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10000.0) / model_size))  

    pe[:, 0::2] = torch.sin(position * div_term)  
    pe[:, 1::2] = torch.cos(position * div_term) 

    return pe.unsqueeze(0)