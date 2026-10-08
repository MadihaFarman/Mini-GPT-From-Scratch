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

@torch.no_grad()
def estimate_loss():
    out = {} 
    model.eval()    # sets the model to evaluation phase
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X , Y = get_batch(split)
            logits , loss = model(X,Y)
            losses[k] =   loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
            targets = targets.view(B*T)
            loss = f.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits , loss = self(idx)
            # focus only on the last time step
            logits = logits[:,-1,:]
            # apply softmax to get prob
            prob = torch.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_sample=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim=1)

        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a pytorch optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        pass

    # sample a batch of data 
    xb, yb = get_batch('train')

    # calculate loss
    logits , loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model

    

    






