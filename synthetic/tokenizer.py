"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from base import Tokenizer, get_stats, merge
import sys
import pickle
import torch
import os
base_path = os.path.join(os.path.dirname(__file__))
base_path = os.path.dirname(base_path)



# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}



    #utf-8 (256 encodings)
    '''
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        sys.exit()
        for i in range(num_merges):
            print(i)
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
    '''
    #65 encoding type
    def train(self, text, vocab_size, verbose=False):
        def get_stats(ids, counts=None):
            """
            Given a list of integers, return a dictionary of counts of consecutive pairs
            Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
            Optionally allows to update an existing dictionary of counts
            """
            counts = {} if counts is None else counts
            for pair in zip(ids, ids[1:]): # iterate consecutive elements
                counts[pair] = counts.get(pair, 0) + 1
            return counts


        def merge(ids, pair, idx):
            """
            In the list of integers (ids), replace all consecutive occurrences
            of pair with the new integer token idx
            Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
            """
            newids = []
            i = 0
            while i < len(ids):
                # if not at the very last position AND the pair matches, replace it
                if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            return newids
    
        chars = sorted(list(set(text)))
        raw_vocab_size = len(chars)
        assert vocab_size >= raw_vocab_size
        num_merges = vocab_size - raw_vocab_size

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        stoi = { ch:i for i,ch in enumerate(chars) }
        raw_encode = lambda s: [stoi[c] for c in s]
        
        #ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        ids = [list(raw_encode(ch)) for ch in text_chunks]
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: [idx] for idx in range(raw_vocab_size)} # idx -> bytes
        print('merges:',num_merges)
        for i in range(num_merges):
            print(i)
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = raw_vocab_size + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        # Save to a file
        with open("tokens_trained.pkl", "wb") as f:
            pickle.dump((merges, vocab), f)

    def train_turbo(self, text, vocab_size, verbose=False):
        from collections import Counter
        import numpy as np
        
        def get_stats_turbo(ids):
            return Counter(zip(ids, ids[1:]))

        def merge_turbo(ids, pair, idx):
            ids = np.array(ids)
            mask = (ids[:-1] == pair[0]) & (ids[1:] == pair[1])
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and mask[i]:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            return new_ids

        def parallel_get_stats(all_ids):
            with ThreadPoolExecutor() as executor:
                results = executor.map(get_stats_turbo, all_ids)
            combined_stats = Counter()
            for result in results:
                combined_stats.update(result)
            return combined_stats

        
        chars = sorted(list(set(text)))
        raw_vocab_size = len(chars)
        assert vocab_size >= raw_vocab_size
        num_merges = vocab_size - raw_vocab_size

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        stoi = { ch:i for i,ch in enumerate(chars) }
        raw_encode = lambda s: [stoi[c] for c in s]
        
        #ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        ids = [list(raw_encode(ch)) for ch in text_chunks]
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: [idx] for idx in range(raw_vocab_size)} # idx -> bytes
        print('merges:',num_merges)
        for i in range(num_merges):
            print(i)
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = raw_vocab_size + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        # Save to a file
        with open("tokens_trained.pkl", "wb") as f:
            pickle.dump((merges, vocab), f)
        

    def load(self, chars):
        with open(f"{base_path}/synthetic/tokens_trained.pkl", "rb") as f:
            merges, vocab = pickle.load(f)
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        self.max_len = max([len(i) for i in self.vocab.values()])
        raw_vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        raw_encode = lambda s: [stoi[c] for c in s]
        self.raw_vocab_size = raw_vocab_size
        self.raw_encode = raw_encode
        self.raw_decode = lambda l: ''.join([itos[i] for i in l])

        voacbt = []
        for k, v in self.vocab.items():
            #tensor_l = [len(v)]
            tensor_v = v[:]
            pad = self.max_len-len(v)
            for i in range(pad):
                tensor_v.append(0)
            tensor_v.append(len(v))
            
            voacbt.append(tensor_v)
        self.voacbt = torch.tensor(voacbt)
        
        

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids, tl_pair=False, chunked=False):
        # given ids (list of integers), return Python string
        chars = []
        tok_to_l = []#token to letter list
        for idx in ids:
            #print('idx:',idx)
            if idx in self.vocab:
                chars_tmp = self.vocab[idx]
                if chunked:
                    chars.append(chars_tmp)
                else:
                    chars.extend(chars_tmp)
                tok_to_l.append([idx,chars_tmp])
            elif idx in self.inverse_special_tokens:
                print('this part of code doest work!')
                chars.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        if chunked:
            text = [self.raw_decode(chunk) for chunk in chars]
        else:
            text = self.raw_decode(chars)
        if tl_pair:
            return tok_to_l
        else:
            return text

    def tok_to_char(self, token_tensor):
        
        #token_tensor = token_tensor.view(-1)
        token_tensor = token_tensor.reshape(-1)
        chars_pad = self.voacbt[token_tensor]
        #print('chars_pad:',chars_pad.shape)
        return chars_pad[:,:8], chars_pad[:,-1:].squeeze().to('cpu')
        '''
        print('do')
        batch_size, seq_size = token_tensor.shape
        # List of lists to hold the character indices for each token in the sequence
        untokens = []
        lengths = []
        max_len = 0
        print('self.vocab:',self.vocab[65])
        for i in range(batch_size):
            for j in range(seq_size):
                token_id = token_tensor[i, j].item()  # Get the token id (0-511)

                # Map token id to corresponding character indices (0-64)
                if token_id < 65:  # If token id is within character range (single character)
                    untokens.append([token_id])  # Token maps to one character
                else:
                    chars = self.vocab[token_id]
                    untokens.append(chars)
                    max_len = max(max_len, len(chars))

                lengths.append(len(untokens[-1]))
        print('done')
        print(self.vocab.items()[:10])
        sys.exit()
        return untokens, lengths, max_len,
        '''

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            #chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_bytes = self.raw_encode(chunk)#.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
