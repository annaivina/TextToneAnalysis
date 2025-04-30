import torch 


class SarcasmClassifier(torch.nn.Module):
  def __init__(self, voc_size, embed_dim, hidden_size, multiplier = 2, is_parent=False, dropout=0.0):
    super().__init__()
    self.is_parent = is_parent

    self.embed = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dim)
    self.lstm_1 = torch.nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)

    if self.is_parent:
      self.lstm_2 = torch.nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
    
    self.linear = torch.nn.Linear(hidden_size * multiplier, 2) #because we have bidirectional lstm
    self.dropout = torch.nn.Dropout(dropout)

  def encode(self, x, lstm):
    x = self.embed(x)
    _, (h_n, _) = lstm(x)
    h_forward = h_n[-2,:,:] #take forward dir last
    h_backward = h_n[-1,:,:]#take backward last
    return torch.cat((h_forward, h_backward), dim=1)


  def forward(self, x: torch.Tensor, y: torch.Tensor = None):
    x_1 = self.encode(x, self.lstm_1)
    
    if self.is_parent and y is not None:
      x_2 = self.encode(y, self.lstm_2)
      x = torch.cat((x_1, x_2), dim=1)
    else:
      x = x_1
    
    
    x = self.dropout(x)
    return self.linear(x)