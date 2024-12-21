import os
import csv
import random
from random import sample
import sys
from tokenizer import RegexTokenizer

#’
dummies = {
'letter_count':
	{
	'Q':
		[
		"How many times does the letter {letter} appear in the word {word}?",
		"In the word {word}, how many occurrences of the letter {letter} are there?",
		"Can you tell me how often the letter {letter} appears in {word}?",
		"What is the count of the letter {letter} in {word}?",
		"How many {letter}s are present in the word {word}?",
		"In {word}, what is the frequency of the letter {letter}?",
		"Find out how many times {letter} is used in {word}.",
		"What is the total number of {letter}s in the word {word}?",
		"How many times does {letter} appear in {word}?",
		"Tell me the number of occurrences of the letter {letter} in {word}.",
		"What's the count of {letter} in {word}?",
		"How many {letter}s does the word {word} contain?",
		"Can you count how many times the letter {letter} appears in {word}?",
		"How often does {letter} show up in the word {word}?",
		"In {word}, how many {letter}s can you find?",
		"What is the number of {letter}s in the word {word}?",
		"How frequently does the letter {letter} occur in {word}?",
		"Can you tell me how many {letter}s are found in {word}?",
		"In the word {word}, how many times does {letter} occur?",
		"How many instances of the letter {letter} are in {word}?",
		]
	,
	'A':
		[
		"The word {word} contains {answer} instances of the letter {letter}.",
		"There are {answer} occurrences of the letter {letter} in {word}.",
		"In {word}, the letter {letter} appears {answer} times.",
		"The total number of {letter}s in {word} is {answer}.",
		"{answer} {letter}s can be found in {word}.",
		"In the word {word}, there are {answer} {letter}s.",
		"The frequency of {letter} in {word} is {answer}.",
		"There are {answer} occurrences of {letter} in {word}.",
		"The word {word} has {answer} {letter}s.",
		"The letter {letter} appears {answer} times in {word}.",
		"In {word}, the letter {letter} appears a total of {answer} times.",
		"There are {answer} occurrences of {letter} in the word {word}.",
		"The letter {letter} occurs {answer} times in {word}.",
		"The word {word} has a total of {answer} {letter}s.",
		"In the word {word}, {letter} appears {answer} times.",
		"There are {answer} {letter}s in {word}.",
		"The count of {letter}s in {word} is {answer}.",
		"In {word}, {letter} shows up {answer} times.",
		"The letter {letter} is found {answer} times in {word}.",
		"In the word {word}, there are {answer} instances of {letter}.",
		]
	},
'reverse_word':
	{
	'Q':
		[
		"What is the reverse of the word {word}?",
		"How would the word {word} look if it were reversed?",
		"Can you provide the word {word} in reverse?",
		"What do you get when you reverse {word}?",
		"How is {word} spelled backwards?",
		"If you reverse the word {word}, what do you get?",
		"What is the backward form of {word}?",
		"Can you give me the reverse of {word}?",
		"How would {word} appear if reversed?",
		"What's the word {word} when flipped backwards?",
		"Please provide the word {word} in reverse.",
		"Can you show me the reversed form of {word}?",
		"What is the word {word} spelled backwards?",
		"How would {word} look if flipped?",
		"What is the backward version of {word}?",
		"If we reversed {word}, what would it be?",
		"Give me the reverse of {word}.",
		"How does {word} look when reversed?",
		"What is the reverse of {word} as a word?",
		"What do you get if {word} is reversed?"
		]
	,
	'A':
		[
		"The reverse of {word} is {answer}.",
		"If you reverse {word}, you get {answer}.",
		"The backward form of {word} is {answer}.",
		"When {word} is reversed, it becomes {answer}.",
		"Spelling {word} backwards gives {answer}.",
		"The word {word} flipped backwards is {answer}.",
		"Reversing {word} results in {answer}.",
		"The reverse of {word} is {answer}, spelled backward.",
		"The word {word} in reverse is {answer}.",
		"If you flip {word}, it turns into {answer}.",
		"The backward version of {word} is {answer}.",
		"Flipping {word} gives {answer}.",
		"When {word} is reversed, it turns into {answer}.",
		"Reversing {word} results in the word {answer}.",
		"The word {word}, when reversed, is {answer}.",
		"If you flip the word {word}, you get {answer}.",
		"When you reverse {word}, it spells {answer}.",
		"Reversing {word} forms {answer}.",
		"Spelling {word} backward gives you {answer}.",
		"The reverse of {word} spells {answer}."
		]
	},
'index_letter':
	{
	'Q':
		[
		"What is the position of the letter {letter} in the word {word}?",
		"Can you tell me where the letter {letter} is located in {word}?",
		"In the word {word}, where does the letter {letter} appear?",
		"Where is the letter {letter} placed in {word}?",
		"Can you find the index of {letter} in {word}?",
		"At what index does the letter {letter} occur in {word}?",
		"How many positions in {word} does the letter {letter} appear?",
		"What is the index of the first {letter} in {word}?",
		"In the word {word}, what's the index of {letter}?",
		"At what position does the letter {letter} occur in {word}?",
		"Where does {letter} appear in {word} for the first time?",
		"Can you point out the position of {letter} in {word}?",
		"Find the index where {letter} appears in {word}.",
		"Where is {letter} in {word}?",
		"What is the first occurrence of {letter} in {word}?",
		"Can you find the letter {letter} in {word} and tell me its position?",
		"What's the position of {letter} in {word}?",
		"In {word}, where is the letter {letter} located?",
		"Tell me the index of {letter} in the word {word}.",
		"Where can I find the letter {letter} in {word}?"
		]
	,
	'A':
		[
		"The letter {letter} is at index {answer} in the word {word}.",
		"In {word}, the letter {letter} appears at position {answer}.",
		"The letter {letter} occurs first at position {answer} in {word}.",
		"At index {answer}, you will find the letter {letter} in {word}.",
		"The letter {letter} is located at position {answer} in the word {word}.",
		"In the word {word}, the first occurrence of {letter} is at index {answer}.",
		"The position of {letter} in {word} is {answer}.",
		"You can find {letter} at index {answer} in {word}.",
		"The first occurrence of {letter} in {word} is at index {answer}.",
		"In {word}, {letter} is at position {answer}.",
		"The letter {letter} is found at index {answer} in the word {word}.",
		"At position {answer} in {word}, the letter {letter} is found.",
		"In the word {word}, the letter {letter} is located at index {answer}.",
		"The letter {letter} appears at position {answer} when you look at {word}.",
		"Position {answer} in {word} corresponds to the letter {letter}.",
		"The index of {letter} in {word} is {answer}.",
		"In the word {word}, the letter {letter} is at index {answer}.",
		"At index {answer}, you can spot {letter} in {word}.",
		"The letter {letter} in {word} is found at position {answer}.",
		"You can find {letter} in {word} at index {answer}."
		]
	},
'swap_letter':
	{
	'Q':
		[
		"What happens when you change the letter {letter1} to {letter2} in {word}?",
		"If you replace {letter1} with {letter2} in {word}, what do you get?",
		"What is the new word when you substitute {letter1} with {letter2} in {word}?",
		"Can you tell me the word that results from changing {letter1} to {letter2} in {word}?",
		"When you swap {letter1} for {letter2} in {word}, what word do you form?",
		"If you change {letter1} to {letter2} in {word}, what's the outcome?",
		"What word do you get if you replace {letter1} with {letter2} in {word}?",
		"When {letter1} is replaced by {letter2} in {word}, what word is formed?",
		"What's the result of changing {letter1} to {letter2} in {word}?",
		"What word do you get if you change {letter1} in {word} to {letter2}?",
		"How does the word {word} change if you replace {letter1} with {letter2}?",
		"If {letter1} is substituted with {letter2} in {word}, what is the new word?",
		"What word do you get by replacing {letter1} with {letter2} in {word}?",
		"Can you figure out the new word when you replace {letter1} with {letter2} in {word}?",
		"What word is formed by changing {letter1} to {letter2} in {word}?",
		"When {letter1} is swapped with {letter2} in {word}, what do you get?",
		"What happens when {letter1} is substituted by {letter2} in {word}?",
		"In {word}, what word do you get if you swap {letter1} for {letter2}?",
		"What's the outcome of swapping {letter1} with {letter2} in {word}?",
		"If you exchange {letter1} with {letter2} in {word}, what word do you form?"
		]
	,
	'A':
		[
		"The word after replacing {letter1} with {letter2} in {word} is {answer}.",
		"When you swap {letter1} for {letter2} in {word}, the resulting word is {answer}.",
		"If you change {letter1} to {letter2} in {word}, the word becomes {answer}.",
		"After replacing {letter1} with {letter2} in {word}, you get {answer}.",
		"The word you get after changing {letter1} to {letter2} in {word} is {answer}.",
		"When you swap {letter1} with {letter2} in {word}, the result is {answer}.",
		"After swapping {letter1} for {letter2} in {word}, the new word is {answer}.",
		"If you replace {letter1} with {letter2} in {word}, you form {answer}.",
		"By changing {letter1} to {letter2} in {word}, you get {answer}.",
		"The word formed by replacing {letter1} with {letter2} in {word} is {answer}",
		"When you replace {letter1} with {letter2} in {word}, the word becomes {answer}",
		"If you exchange {letter1} with {letter2} in {word}, the new word is {answer}",
		"By substituting {letter1} with {letter2} in {word}, you get {answer}.",
		"The word you get by replacing {letter1} with {letter2} in {word} is {answer}.",
		"The result of swapping {letter1} with {letter2} in {word} is {answer}",
		"If you change {letter1} to {letter2} in {word}, the new word is {answer}.",
		"By replacing {letter1} with {letter2} in {word}, the resulting word is {answer}.",
		"When {letter1} is swapped with {letter2} in {word}, the new word formed is {answer}.",
		"What word do you get after changing {letter1} to {letter2} in {word}? It's {answer}.",
		"If {letter1} is substituted for {letter2} in {word}, you end up with {answer}.",
		]
	}

}

raw_vocab = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¢', '£', '§']



#dummies removoe all avfter {answer}
for q_type in dummies:
	dummies[q_type]['A'] = [a[:a.find("{answer}")+8] for a in dummies[q_type]['A']]

'''
qa_max = {'q':0,'a':0}
add_q = len('Question: - ')
add_a = len('Answer - ')
mid = len(' \n')
for type_task in dummies:
	for q in dummies[type_task]['Q']:
		if len(q) > qa_max['q']:
			qa_max['q'] = len(q)
	for a in dummies[type_task]['A']:
		if len(a) > qa_max['a']:
			qa_max['a'] = len(a)
qa_max['q']+=add_q
qa_max['a']+=add_a
max_text = qa_max['q']+add_q+qa_max['a']+add_a+mid
print(qa_max)
print('max_text:',max_text)
sys.exit()
'''

def load_words():
	words = []
	# Open the CSV file
	with open(f'{os.path.dirname(os.path.realpath(__file__))}/unigram_freq.csv', mode='r') as file:
		reader = csv.reader(file)
		
		# Loop through the first 10 rows and print them
		for i, row in enumerate(reader):
			if i < 50000:
				#print(row)
				word = row[0]
				if 5 <= len(word) <= 17:
					words.append(word)
			else:
				break
	return words[1:]

class QAdataset():

	def __init__(self, val_split=0.2, split='train'):
		self.all_words = load_words()
		self.dummies = dummies
		self.dummy_inserts = {'q':['Qestion: ','Qestion:\n','Qestion: \n','Q: ', 'Q\n','','Question: - '],'a':['Answer: ','Answer: \n','Answer:\n','Answer - ','A: ', 'A\n','','A - '],'connect':[' ','\n',' \n']}
		self.seed = 42
		random.seed(self.seed)
		random.shuffle(self.all_words)
		self.val_split = int(len(self.all_words)*val_split)
		self.val_words = self.all_words[:self.val_split]
		self.train_words = self.all_words[self.val_split:]
		self.split(split=split)

	def split(self, split='train'):
		if split == 'train':
			self.words_for_use = self.train_words
		elif split == 'val':
			self.words_for_use = self.val_words
		else:
			raise Exception(f"split with value '{split}' is unknown ")

	def count(self):
		question_wrap = sample(self.dummies['letter_count']['Q'], 1)[0]
		answer_wrap = sample(self.dummies['letter_count']['A'], 1)[0]
		count = random.randint(1,4)
		if random.randint(0,20)==0:
			count+=1
		while True:
			word = sample(self.words_for_use, 1)[0]
			max_n = 0
			for l in set(word):
				x = word.count(l)
				if max_n < x:
					max_n = x
					letter = l

			if max_n>=count:
				answer = max_n
				break

		text = sample(self.dummy_inserts['q'],1)[0] + question_wrap + sample(self.dummy_inserts['connect'],1)[0] + sample(self.dummy_inserts['a'],1)[0] + answer_wrap
		text = text.replace('{letter}',letter).replace('{word}',word)
		a_indx = text.find('{answer}')
		text = text.replace('{answer}',str(answer))
		return text, a_indx

	def reverse(self):
		question_wrap = sample(self.dummies['reverse_word']['Q'], 1)[0]
		answer_wrap = sample(self.dummies['reverse_word']['A'], 1)[0]
		word = sample(self.words_for_use, 1)[0]
		answer = word[::-1]
		text = sample(self.dummy_inserts['q'],1)[0] + question_wrap + sample(self.dummy_inserts['connect'],1)[0] + sample(self.dummy_inserts['a'],1)[0] + answer_wrap
		text = text.replace('{word}',word)
		a_indx = text.find('{answer}')
		text = text.replace('{answer}',str(answer))
		return text, a_indx

	def indexing(self):
		question_wrap = sample(self.dummies['index_letter']['Q'], 1)[0]
		answer_wrap = sample(self.dummies['index_letter']['A'], 1)[0]
		word = sample(self.words_for_use, 1)[0]
		letter = sample(list(word), 1)[0]
		answer = word.find(letter)
		text = sample(self.dummy_inserts['q'],1)[0] + question_wrap + sample(self.dummy_inserts['connect'],1)[0] + sample(self.dummy_inserts['a'],1)[0] + answer_wrap
		text = text.replace('{letter}',letter).replace('{word}',word)
		a_indx = text.find('{answer}')
		text = text.replace('{answer}',str(answer))
		return text, a_indx

	def swap(self):
		question_wrap = sample(self.dummies['swap_letter']['Q'], 1)[0]
		answer_wrap = sample(self.dummies['swap_letter']['A'], 1)[0]
		word = sample(self.words_for_use, 1)[0]
		setw = set(word)
		letter1 = sample(sorted(setw), 1)[0]
		let_indx = word.find(letter1)
		setw.remove(letter1)
		letter2 = sample(sorted(setw), 1)[0]
		answer = word[:let_indx] + letter2 + word[let_indx + 1:]
		text = sample(self.dummy_inserts['q'],1)[0] + question_wrap + sample(self.dummy_inserts['connect'],1)[0] + sample(self.dummy_inserts['a'],1)[0] + answer_wrap
		text = text.replace('{letter1}',letter1).replace('{letter2}',letter2).replace('{word}',word)
		a_indx = text.find('{answer}')
		text = text.replace('{answer}',str(answer))
		return text, a_indx

	def get_batch(self, n, tokenizer=None, split='train'):
		self.split(split=split)
		batch = []
		while len(batch) < n:
			functions = [self.swap, self.indexing, self.reverse, self.count]
			text, a_indx = random.choice(functions)()
			a_indx = a_indx
			if tokenizer == None:
				batch.append([text, a_indx]) 
			else:
				tokens = tokenizer.encode(text)
				tok_to_let = tokenizer.decode(tokens, tl_pair=True)
				counter = 0
				for ti, (token, letters) in enumerate(tok_to_let):
					counter+=len(letters)
					if counter>a_indx:# > is correct! >= is worng!
						a_indx = ti
						break

				batch.append([tokens, a_indx])

		return batch



'''
qa = QAdataset()

tokenizer = RegexTokenizer()
tokenizer.load(raw_vocab)
tokens, ans_ind = qa.get_batch(1, tokenizer=tokenizer)[0]


untok_row = tokenizer.decode(tokens,tl_pair=True)
text= tokenizer.decode(tokens)
splits = []
for c in untok_row:
	idx, chars_tmp = c
	seq_chunk = tokenizer.raw_decode(chars_tmp)
	splits.append(seq_chunk)
print(text)
print(splits)
'''

#print(tokens)
#text= tokenizer.decode(tokens)
#print(text)
#p1, p2 = tokens[:ans_ind], tokens[ans_ind:]
#print(p1)
#print(p2)
#sys.exit()
#p1, p2 = tokenizer.decode(p1), tokenizer.decode(p2)
#print(list(p1))
#print(list(p2))



#print()
#print(batch)
#print(text_back)
#max after 1000000 is 94
#print(text)
#tokens = tokenizer.encode(text)
#print(tokens)
#docoded = tokenizer.decode(tokens)
#print(docoded)

#print(len(batch), max(batch))


















