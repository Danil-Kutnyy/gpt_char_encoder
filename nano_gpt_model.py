import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import sys
import tiktoken
from tokenizer import RegexTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import pickle
import os
base_path = os.path.join(os.path.dirname(__file__))

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#device = 'cpu'
print('device:',device)
eval_iters = 128
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.15
# ------------

torch.manual_seed(42)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(f'{base_path}/shakespeare.txt', 'r', encoding='utf-8') as f:
	text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
raw_vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

'''
gpt_vanila_tokenizer = tiktoken.encoding_for_model('davinci')
print(len(text))
tokens = gpt_vanila_tokenizer.encode(text)
tokens = sorted(list(set(tokens)))
print(len(tokens), tokens[:10],tokens[-10:])

class Tokenizer():
	def __init__(self, text=text, encoder=gpt_vanila_tokenizer):
		tokens = gpt_vanila_tokenizer.encode(text)
		token = set()
'''

tokenizer = RegexTokenizer()
#tokenizer.train(text, vocab_size=512)
tokenizer.load(text)
tokenized_text = tokenizer.encode_ordinary(text)
#tokenized_text = encode(text)
vocab_size = 512




# Train and test splits
data = torch.tensor(tokenized_text, dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
	# generate a small batch of data of inputs x and targets y
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y


@torch.no_grad()
def estimate_loss(model):
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

def evaluate_models(models):
    #Evaluate loss for multiple models on the same test and validation data.
    #Args: models (list): List of models to evaluate.
    #Returns: A list of dictionaries where each dictionary contains the average 
    #loss for 'train' and 'val' splits for a corresponding model.
    results = [{} for _ in models]  # One dictionary per model

    for split in ['train', 'val']:
        for k in range(eval_iters):
            # Generate a single batch to use for all models
            X, Y = get_batch(split)

            for i, model in enumerate(models):
                model.eval()
                with torch.no_grad():
                    logits, loss = model(X, Y)
                    if split not in results[i]:
                        results[i][split] = []
                    results[i][split].append(loss.item())
    
    # Average the losses for each model and split
    for result in results:
        for split in ['train', 'val']:
            result[split] = sum(result[split]) / len(result[split])

    return results
   
'''
class SequenceToVector(nn.Module):
	def __init__(self, input, hidden, out):
		super(SequenceToVector, self).__init__()
		self.lstm = nn.LSTM(input_size=input, hidden_size=hidden, batch_first=True)  # Encode sequence
		self.fc = nn.Linear(hidden, out)  # Map to 512 dimensions

	def forward(self, x):
		_, (h_n, _) = self.lstm(x)  # h_n: final hidden state
		output = self.fc(h_n[-1])   # Map to 512-length vector
		return output
'''
class SequenceToVector(nn.Module):
	def __init__(self, input_size, hidden_size, out_size):
		super(SequenceToVector, self).__init__()
		# Embedding layer: Transform one-hot vectors to dense representations
		self.embedding = nn.Embedding(input_size, hidden_size)  # 65 possible tokens
		self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
		self.fc = nn.Linear(hidden_size, out_size)  # Output to 512 dimensions

		for name, param in self.named_parameters():
			if 'lstm.bias' in name:
				# LSTM biases are split into four chunks (input, forget, cell, output gates)
				n = param.size(0)
				forget_gate_bias_start = n // 4
				forget_gate_bias_end = n // 2
				param.data[forget_gate_bias_start:forget_gate_bias_end].fill_(1.0)

	def forward(self, idx):
		#print('start')
		#t = time.time()
		# x: padded one-hot tensor of size (batch_size, seq_len, 65)
		# Convert one-hot encoded tokens to dense vectors using the embedding layer
		max_len = tokenizer.max_len
		#print('max_len_totoal:',max_len)
		tokenizer.voacbt = tokenizer.voacbt.to(device)
		untokens, lengths = tokenizer.tok_to_char(idx, )
		#print(untokens.shape)
		#print(lengths.shape)
		#sys.exit()
		#print(time.time()-t)
		#t=time.time()
		#untokens = [torch.tensor(seq) for seq in untokens]
		#untokens = [torch.tensor(seq) for seq in untokens]
		#print(time.time()-t)
		#t=time.time()
		#new_len = [len(seq) for seq in untokens]
		#print(time.time()-t)
		#t=time.time()
		#padded_tensor = pad_sequence(untokens, batch_first=True, padding_value=0)
		#print(time.time()-t)
		#t=time.time()
		#padded_tensor = padded_tensor.to(device)
		#print(time.time()-t)
		#t=time.time()

		#print(untokens.shape)
		embedded = self.embedding(untokens)
		#print(embedded.shape)
		#print('lengths:',lengths.shape)
		#print(time.time()-t)
		#t=time.time()
		# Pack the padded sequence for efficient LSTM processing
		#print('embedded:',embedded.shape)
		



		#real versions, but due to mps limitation
		#packed_input = pack_padded_sequence(embedded, lengths.to('cpu'), batch_first=True, enforce_sorted=False)#.to('cpu')
		#packed_output, (h_n, _) = self.lstm(packed_input)
		#output = self.fc(h_n[-1].to(device))  # Map to 512-length vector
		
		#mps problems:
		# Split the tensor into two halves
		size = embedded.size(0)
		seq_len = embedded.size(1)  # 8
		hidden_size = embedded.size(2)  # 128
		first_half = embedded[:size//2]  # Shape: [8192, 8, 128]
		second_half = embedded[size//2:]  # Shape: [8192, 8, 128]

		# Assuming lengths for both halves are the same, for simplicity (adjust if needed)
		lengths_first_half = lengths[:size//2]
		lengths_second_half = lengths[size//2:]

		# Pack the sequences for both halves
		packed_first_half = pack_padded_sequence(first_half, lengths_first_half, batch_first=True, enforce_sorted=False)
		packed_second_half = pack_padded_sequence(second_half, lengths_second_half, batch_first=True, enforce_sorted=False)
		#print(time.time()-t)
		#t=time.time()
		# Now you pass each half through the LSTM (assume self.lstm is your LSTM layer)
		packed_output_first_half, (h_n_first_half, _) = self.lstm(packed_first_half)
		packed_output_second_half, (h_n_second_half, _) = self.lstm(packed_second_half)
		#print(time.time()-t)
		#t=time.time()
		# Combine the hidden states from both halves
		# You can concatenate them (along the hidden state dimension) or add them, depending on your use case
		#sys.exit()
		combined_hidden_state = torch.cat((h_n_first_half[-1], h_n_second_half[-1]), dim=0)
		#print('combined_hidden_state:',combined_hidden_state.shape)
		#sys.exit()
		output = self.fc(combined_hidden_state) 
		'''
		output_tensor = output
		#print(time.time()-t)
		#t=time.time()
		#print(output.view(batch_size, block_size, -1).shape)
		#sys.exit()
		
		#print(time.time()-t)
		#print('done')
		#sys.exit()
		# Calculate statistics
		mean = output_tensor.mean().item()
		std = output_tensor.std().item()

		print(f"Mean: {mean:.4f}")
		print(f"Std: {std:.4f}")

		# Convert to NumPy for visualization
		output_np = output_tensor.cpu().numpy()

		# Plot histograms for each feature (or trails)
		plt.figure(figsize=(10, 6))
		plt.hist(output_np.flatten(), bins=50, color='blue', alpha=0.7)
		plt.title("Histogram of Layer Outputs")
		plt.xlabel("Output Value")
		plt.ylabel("Frequency")
		plt.grid(True)
		plt.show()
		'''
		return output.view(batch_size, block_size, -1)


class Head(nn.Module):
	""" one head of self-attention """

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# input of size (batch, time-step, channels)
		# output of size (batch, time-step, head size)
		B,T,C = x.shape
		k = self.key(x)   # (B,T,hs)
		q = self.query(x) # (B,T,hs)
		# compute attention scores ("affinities")
		wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		wei = F.softmax(wei, dim=-1) # (B, T, T)
		wei = self.dropout(wei)
		# perform the weighted aggregation of the values
		v = self.value(x) # (B,T,hs)
		out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
		return out

class MultiHeadAttention(nn.Module):
	""" multiple heads of self-attention in parallel """

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(head_size * num_heads, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out

class FeedFoward(nn.Module):
	""" a simple linear layer followed by a non-linearity """

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout),
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
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x

class GPTLanguageModel(nn.Module):

	def __init__(self, arch_type='original'):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd) # final layer norm
		self.lm_head = nn.Linear(n_embd, vocab_size)
		self.arch_type = arch_type
		if self.arch_type == 'char_encoder_add':
			self.char_encode = SequenceToVector(raw_vocab_size, n_embd, n_embd)

		# better init, not covered in the original GPT video, but important, will cover in followup video
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx, targets=None,fc=False):
		B, T = idx.shape

		# idx and targets are both (B,T) tensor of integers
		tok_emb = self.token_embedding_table(idx) # (B,T,C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
		x = tok_emb + pos_emb # (B,T,C)
		if self.arch_type == 'char_encoder_add':
			char_emb = self.char_encode(idx)
			x = x + char_emb


		x = self.blocks(x) # (B,T,C)
		x = self.ln_f(x) # (B,T,C)
		logits = self.lm_head(x) # (B,T,vocab_size)
		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop idx to the last block_size tokens
			idx_cond = idx[:, -block_size:]
			# get the predictions
			logits, loss = self(idx_cond)
			# focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # (B, C)
			# sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			# append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx



