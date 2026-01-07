# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

#hyperparameters 
batch_size = 32 #32 independent sequences will we process in parallel
block_size = 8 #the model will look to 8 previous characters to predict the next one 
max_iters = 5000 #update the weights 5000 times
eval_interval = 500 #every 500 steps checks how accurate the model is
learning_rate = 1e-3 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #when we check accuracy, we test on 200 random batches to get a good average
n_embd = 32 #the size of a vector representing each charachter (32 numbers per letter)
torch. manual_seed(1337)


#!wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt
#uploaded dataset from internet

try:
   with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
except FileNotFoundError:
   print("Error: input.txt not found")
   text = "Error loading data. This is a placeholder string to prevent crash."
  #code opens text and reads it


#all unique characters in the text
chars = sorted(list(set(text))) #set automatically removes duplicates, and creates a set of all unique characters in the text, list creates list like this ['b','a','n'] and then orders it
vocab_size = len(chars) #outputs lengths of the chars


#create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}  #creates a dictionary to convert charachters to numbers
itos = {i:ch for i,ch in enumerate(chars)} #creates a dictionary to convert numbers to charachters

encode = lambda s: [stoi[c] for c in s] #encoder take a string, output a list of integers, h=0, a=1
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string decode[0,1,2,3] -> hii there


#encoded the entire text dataset and stored it into a torch.tensor

data = torch.tensor(encode(text), dtype=torch.long) #torch tensor is a multi-dimensional array, torch.long = 64-bit integers
#splitted data into train and validation sets
n = int(0.9*len(data)) #first 90% train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split): 
  #generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data #if split is train it will use train else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) 
  x = torch.stack([data[i:i+block_size] for i in ix]) #if the text is hello, and block_size is 4, x might be hell
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #the targets are shifted by one. For "hell", the target is "ello". We want the model to predict 'e' after 'h', 'l' after 'e'
  x,y = x.to(device), y.to(device)
  return x,y

@torch.no_grad() #tells pytorch to not calculate gradients
def estimate_loss(): #runs the model on both training and validation data, we can see loss
   out = {}
   model.eval()
   for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
         X, Y = get_batch(split)
         logits, loss = model (X, Y)
         losses[k] = loss.item()
      out[split] = losses.mean()
   model.train()
   return out     

class Head (nn.Module):
   "one head of self-attention"

   def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias = False)
      self.query = nn.Linear(n_embd, head_size, bias = False)
      self.value = nn.Linear(n_embd, head_size, bias = False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


   def forward(self,x):
      B,T,C = x.shape
      k = self.key(x) #Key: What do I contain (I am a vowel)
      q = self.query(x) #Query: What am I looking for? (I am looking for a consonant)
      #compute attention scores ("affinities")
      wei = q @ k.transpose(-2,-1) * C ** -0.5 #(B,T,C) @ (B,C, T) --> (B,T,T)
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #masking, don't look at the future
      wei = F.softmax(wei, dim=-1) #Normalize probabilities sum to 1
      v = self.value(x) 
      out = wei @ v #output info
      return out


class MultiHeadAttention(nn.Module): #running several Head's at once. One head might focus on grammar, another on rhymes, another on names. Then results will be concatenated
   

   def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(0.2)
   def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)
      return out
   

class FeedForward(nn.Module): 
   "a simple linear layer followed by a non-linearity"

   def __init__(self, n_embd):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd, 4 * n_embd),
         nn.ReLU(),
         nn.Linear(4 * n_embd, n_embd),
      )

   def forward(self, x):
      return self.net(x)
   

class Block(nn.Module):
   "Transformer block: communication fillowed by computation"

   def __init__(self, n_embd, n_head):
      #n_embd: embedding dimension, n_head: the number of heads we would like
      super().__init__()
      head_size = n_embd//n_head
      self.sa = MultiHeadAttention(n_head, head_size)
      self.ffwd = FeedForward(n_embd)
      self.ln1 = nn.LayerNorm(n_embd)
      self.ln2 = nn.LayerNorm(n_embd)

  
   def forward(self,x):
       x = x + self.sa(self.ln1(x))
       x = x+ self.ffwd(self.ln2(x))
       return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
    #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd) 
        self.blocks = nn.Sequential(
           Block(n_embd, n_head=4),
           Block(n_embd, n_head=4),
           Block(n_embd, n_head=4),
           nn.LayerNorm(n_embd),
        )
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) #4 heads of 8-dimensional self-attention
        #self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward (self, idx, targets=None):
        B, T = idx.shape 

    #idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # identity to the token (Is it a or b?)
        pos_emb = self.position_embedding_table(torch.arange(T, device= device)) #Position of the token ( Is it 1st or 5th)
        x = tok_emb + pos_emb 
        x = self.blocks(x) #go through the transformers blocks 
        #x = self.sa_heads(x)
        #x = self.ffwd(x) 
        logits = self.lm_head(x) #final layer to guess the next character
#32*65, 2D array; calculated loss
        if targets is None:
          loss = None
        else:
           B, T, C = logits.shape
           logits = logits.view(B*T, C)
           targets = targets.view(B*T)
           loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): #idx - the input which is a PyTorch tensor of token indices representing the starting prompt or context. It has a shape of (B,t), max_new_tokens - maximum number of new tokens the function should generate
      #idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
        #crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        #get the predictions
        logits, loss= self(idx_cond) #this line performs a forward pass, it feeds the current sequence of tokens(idx) into the model
        #focus only on the last time step, focuses at the very last token
        #logits - the model's raw, unnormalized output scores.
        logits = logits[:, -1, :] #becomes (B,C)
        #apply softmax to get probabilities, between 0 and 1; all values in the distribution sum to 1; now we have the probability for every [ossible next token]
        probs = F.softmax(logits, dim=-1) #(B,C)
        #sample from the distribution; instead of picking the token with the highest probability torch.multinomial treats the probabilities like a weighted die. It's more likely to pick high-probability tokens, but it gives lower-probsbility tokens a chance, leading to more interesting and varied output
        idx_next = torch.multinomial(probs, num_samples=1) #(B,1) #num_samples=1- we are only sampling one next token
        #append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) #(B, T+1) #concatenates(joins) the newly sampled token(idx_next) to the end of our current sequence(idx)
      return idx

model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for iter in range(max_iters):

    if iter % eval_interval == 0:
       losses = estimate_loss()
       print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  #sample a batch of data
    xb, yb = get_batch('train')

  #evaluate the loss
    logits, loss= model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


