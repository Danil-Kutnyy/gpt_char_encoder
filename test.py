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
from nano_gpt_model import GPTLanguageModel, get_batch  # Assuming your model class is in this file
base_path = os.path.join(os.path.dirname(__file__))


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
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

def evaluate_models(models):
	#Evaluate loss for multiple models on the same test and validation data.
	#Args: models (list): List of models to evaluate.
	#Returns: A list of dictionaries where each dictionary contains the average 
	#loss for 'train' and 'val' splits for a corresponding model.
	results = [{} for _ in models]  # One dictionary per model
	#result = {0:{val:[],train:[]},1:{val:[],train:[]}}

	for split in ['train', 'val']:
		for k in range(eval_iters):
			print(k)
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
	avrg = {}
	for i, result in enumerate(results):
		avrg[i] = {}
		for split in ['train', 'val']:
			avrg[i][split] = sum(result[split]) / len(result[split])

	return results, avrg
   


# Define constants
#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
#num_batches = 128
#batch_size = 64
#eval_iters = num_batches  # Evaluate one batch at a time
eval_split = 'val'  # Assume evaluation on validation data

# Paths to saved weights
original_model_path = f"{base_path}/model_original_weights.pth"
char_encoder_model_path = f"{base_path}/model_seqtoken_weights.pth"

# Load models
model_original = GPTLanguageModel(arch_type='original').to(device)
model_char_encoder_add = GPTLanguageModel(arch_type='char_encoder_add').to(device)

model_original.load_state_dict(torch.load(original_model_path, map_location=device))
model_char_encoder_add.load_state_dict(torch.load(char_encoder_model_path, map_location=device))


# Calculate losses for both models
#losses_original = evaluate_model_loss(model_original, num_batches, batch_size, eval_split)
#losses_char_encoder_add = evaluate_model_loss(model_char_encoder_add, num_batches, batch_size, eval_split)
results, avrg = evaluate_models([model_original, model_char_encoder_add])
print('avrg:',avrg)

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, eval_iters + 1), results[0]['val'], label='Original Model', marker='o', linestyle='--')
plt.plot(range(1, eval_iters + 1), results[1]['val'], label='Char Encoder Add Model', marker='x', linestyle='-')
plt.title('Loss Comparison Across Batches')
plt.xlabel('Batch Index')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()






