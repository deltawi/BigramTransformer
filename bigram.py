from utils.utils import xml_to_pandas, find_device
import torch 
import torch.nn as nn
from torch.nn import functional as F
import math

# Data Section
df_from_file = xml_to_pandas("./Sabanews_utf_8.xml")
text = ''.join(df_from_file['Text'].values)
print(text[:9])
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab_size = ", vocab_size)
# Create simple tokenizer
char_toid = { ch:i for i,ch in enumerate(chars)}
id_tochar = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [char_toid[c] for c in s]
decode = lambda ids: [id_tochar[id] for id in ids]

# Model parameters section
batch_size=32
block_size=32
max_iters = 5000
eval_interval = 300
learning_rate = 1e-4
device = find_device(debug=True)
eval_iters = 200
n_embed = 32
dropout = 0.2

torch.manual_seed(2024)

# Train and test split 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttentionHead(nn.Module):

    def __init__(self, n_embed, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (Batch, Time, Head_size)
        q = self.query(x) # (Batch, Time, Head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (Batch, Time, Head_size) @ (Batch, Head_size, Time) -> (Batch, Time, Time)
        wei = wei.masked_fill(self.tril[:T, :T] ==0 , float('-inf')) # Triangular masking
        wei = F.softmax(wei, dim=-1) # Softmax on the rows of each batch

        v = self.value(x) # (Batch, Time, Head_size)

        out = wei @ v # (Batch, Time, Time) @ (Batch, Time, Head_size) -> (Batch, Time, Head_size)

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embed, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), # From the paper Attention is all you need they multiply the fork dimension by 4
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed), # Going back to the original dimension n_embed   
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):

    def __init__(self, n_embed, head_size):
        super().__init__()
        num_heads = n_embed // head_size
        self.attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln_norm1 = nn.LayerNorm(n_embed)
        self.ln_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln_norm1(x))
        x = x + self.ffwd(self.ln_norm2(x))

        return x
    
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.attention_blocks = nn.Sequential(
            AttentionBlock(n_embed, head_size=4),
            AttentionBlock(n_embed, head_size=4),
            AttentionBlock(n_embed, head_size=4),
            AttentionBlock(n_embed, head_size=4)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.n_embed = n_embed

    def forward(self, idx, target=None):
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx) # (Batch, Time, n_embed). Channel is coming from the Embedding layer for each one of the input
        pos_embed =self.position_embedding_table(torch.arange(T, device=device)) # (Time, n_embed)
        x = token_embed + pos_embed # (Batch, time, n_embed)
        x = self.attention_blocks(x)
        logits = self.lm_head(x) # (Batch, Time, vocab_size)

        if target is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # This squach the two first dimensions B and T so that C is the second to comply with cross entropy
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target) # Cross entropy would expect (Batch, Channel, Time)
            return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:] # Selecting only the last x chacaters (max = block_size) -> (Batch, Time=block_size)
            logits, loss = self(idx_cond) # Get the prediction

            logits = logits[:, -1, :] # Focus only on the last timestep => (B, C) because T becomes 1

            probs = F.softmax(logits, dim=-1)  # Apply softmax converts to probabilites across the Channels

            idx_next = torch.multinomial(probs, num_samples=1) # We sample from the prob distribution to get the next token (B, 1)

            idx = torch.cat((idx, idx_next), dim=1) # Concatenate the token with the existing sequence (B, T+1)
        
        return idx

# Initiate the model
model = BigramLanguageModel(vocab_size=vocab_size, n_embed=n_embed)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    if iter % eval_interval==0:
        evaluation = estimate_loss(model)
        print(f"Train loss {evaluation['train']}, Validation loss {evaluation['val']}")
    xb, yb = get_batch("train")
    #print(''.join(decode(xb[0].tolist())))
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(''.join(decode(model.generate(context, max_new_tokens=100)[0].tolist()))) 

