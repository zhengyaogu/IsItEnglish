import torch
from torch import nn
from data import Vocab

class CNNForCorruptionClassification(nn.Module):

    def __init__(self, vocab: Vocab,
                 embed_dim = 512,
                 out_channels = 64,
                 window_sizes = [4, 8, 16]) -> None:
        super(CNNForCorruptionClassification, self).__init__()
        self.vocab = vocab
        self.embed_dim = 512
        self.out_channels = 64
        self.window_sizes = window_sizes

        self.embedding = nn.Embedding(num_embeddings=len(self.vocab),
                                      embedding_dim=embed_dim)
        self.convs = [nn.Conv2d(in_channels=2,
                               out_channels=out_channels,
                               kernel_size=(n, embed_dim),
                               stride=1)
                      for n in window_sizes]
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features = out_channels * len(window_sizes),
                                out_features = 2)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        input: a N * 2 * T tensor of input ids, where N is the batch size, T is the longest sequence
        '''
        nextout = self.embedding(input) # output: N * 2 * T * D, D is the embedding dimension
        conv_outputs = [conv(nextout).squeeze(-1) for conv in self.convs] #each element has dimensions N * C_out * T'
                                                                          # T' may varies element by element
        pooled_conv_outputs = [t.max(dim=-1)[0] for t in conv_outputs] # each element has dimensions N * C_out
        pooled_output = torch.cat(pooled_conv_outputs, dim=-1)
        nextout = self.dropout(pooled_output)
        nextout = self.linear(nextout)
        return nextout

        
