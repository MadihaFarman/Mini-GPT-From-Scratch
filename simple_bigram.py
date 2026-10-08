import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-2
max_iters = 3000
eval_iters = 200
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------
 
torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# unique characters that occur in the text 
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenization : create a mapping from string to integer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# tokenize the whole data
data = torch.tensor(encode(text) , dtype=torch.long)

# train and validation split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading 
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)- block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x , y

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()

        # each token directly reads next token logits from a lookup table
        self.token_embedding_model = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits = self.token_embedding_model(idx)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets
    






