import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-3
max_iters = 5000
eval_iters = 200
eval_interval = 500
n_embd = 32
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
        losses = torch.zeros(eval_iters)  # tensor of 200
        for k in range(eval_iters):  # 200 iterations
            X , Y = get_batch(split)
            logits , loss = model(X,Y)
            losses[k] =   loss.item()
        out[split] = losses.mean()
    model.train()   # sets the model back to training phase
    return out

class Head(nn.Module):
    """One head of self attention"""
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))   # tril is not included in model params, so we have to assign it to model using register_buffer


    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5   # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei,dim=-1)   # (B,T,T)

        # perform weighted aggregation of values
        v = self.value(x)   # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) ----> (B,T,C)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)

    def forward(self,x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)
      return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
        )

    def forward(self, x):
        return self.net(x)
        
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
       

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # each token directly reads next token logits from a lookup table

        self.token_embedding_model = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4)
        )
        self.lm_head = nn.Linear(n_embd,vocab_size)   # to go from token emb (B,T,C(n_embd)) to logits (B,T,C(vocab_size)) we need a linear layer

        # under the hood : # For each token vector of length n_embd:
                           # output_vector = token_vector @ weight_matrix.T + bias

    def forward(self,idx,targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_model(idx)   # (B,T,C)  ---- (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))  # (T,C)
        x = tok_emb + pos_emb   # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        logits = self.lm_head(x)   # (B,T,C)  ---- C = vocab_size

        #  Because toke_emb lives in a hidden embedding space of size n_embd, it cannot be directly used to compute loss or sample characters—we need scores for every token in our vocabulary (vocab_size).self.lm_head (Language Model Head) is a linear projection layer (y = xW^T + b) that maps each n_embd-dimensional vector back to a vocab_size-dimensional vector of raw logits.

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens bcz our position emb table has size of only block_size
            idx_cond = idx[:,-block_size:]
            # get predictions
            logits , loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:]
            # apply softmax to get prob
            prob = torch.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim=1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data 
    xb, yb = get_batch('train')

    # calculate loss
    logits , loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model

context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))
