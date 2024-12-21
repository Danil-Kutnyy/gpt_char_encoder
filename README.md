Code here was not prepared for review or use. However, if you need to make sense of it for some reason, 
I prepared a little guide, please contact me, I will be pleased to help.

Email: Danil.kutny@gmail.com
Telegram: @DanilKutny

Required files:
Some additional files are required. This is dateset and its tokenized versions, they are too big to be uploaded to 
GitHub, therefore they are available on google drive:
https://drive.google.com/drive/folders/1d6TmwmeeowA2HxZg83J0FIWYhyJZQPpB?usp=share_link
You need to place them manually, each 3 of them:
1. **tokenized_books.pkl** - tokenized versions of books. Place in main folder
2. **gutenberg/** - folder with dataset of books, place in main folder
3. **tokenized_books_chunked.pkl** - these are chunked tokenized books, it is important for fine-tuning and testing on character-specific tasks. It should be placed inside /synthetic folder. 

Brief Guide:
**/nano_gpt_model** - main file, that load dataset, tokenizer, create models and train them.
**/plot** - will show results of pre-training languages modeling stage, same results as in the paper
All **.pth** files are saved parameters of trained models and optimizers, if you wish to test already trained models
**/synthetic** - contains additional codebase for character-level task fine-tuning. Include a separate character-level manipulation tasks and training on them + results.
**/synthetic/synthetic_data** - creates dataset with character based manipulations
**/synthetic/nano-gpt** - load such dataset and fine-tune models 
**/synthetic/plot** - will show results of fine-tuning on those tasks, same results as in the paper

Additional note:
Tokenizer is already trained as used part of dataset tokenized. But you can do it again 
if you wish with tokenizer.train() method. Be careful as some texts in dataset contain rare characters like Ꞣ. 
They are not worthy to be used for small tokenizer. In my settings such texts were the
