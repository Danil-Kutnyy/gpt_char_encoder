The code here was not prepared for review or general use. However, if you need assistance understanding it, I have prepared a brief guide. Please feel free to contact me—I will be happy to help:


Email: Danil.kutny@gmail.com


Telegram: @DanilKutny


### Required Files
Some additional files are required for this project. These files are too large to be uploaded to GitHub, so they are available on Google Drive:  
**[Google Drive link](https://drive.google.com/drive/folders/1d6TmwmeeowA2HxZg83J0FIWYhyJZQPpB?usp=share_link)**  

You will need to manually place the following files in their appropriate locations:

1. **`tokenized_books.pkl`**  
   - Tokenized versions of books.  
   - Place this file in the **main folder**.

2. **`gutenberg/`**  
   - Folder containing the dataset of books.  
   - Place this folder in the **main folder**.

3. **`tokenized_books_chunked.pkl`**  
   - Chunked tokenized books. This is essential for fine-tuning and testing character-specific tasks.  
   - Place this file inside the **`/synthetic`** folder.

---

### Brief Guide
#### Main Folder
1. **`/nano_gpt_model`**  
   - This is the main file that:  
     - Loads the dataset and tokenizer.  
     - Creates the models.  
     - Trains the models.

2. **`/plot`**  
   - Displays the results of the pre-training language modeling stage.  
   - The results will be the same as reported in the paper.

3. **`.pth` files**  
   - These are saved parameters of trained models and optimizers.  
   - Use them to test pre-trained models.

#### Synthetic Folder
1. **`/synthetic`**  
   - Contains the additional codebase for fine-tuning on character-level tasks.  
   - Includes separate tasks for character-level manipulations and their training.  
   - Also contains scripts for evaluating results.

2. **`/synthetic/synthetic_data`**  
   - Creates datasets with character-based manipulations.

3. **`/synthetic/nano-gpt`**  
   - Loads datasets for character-based tasks.  
   - Fine-tunes the models on these datasets.

4. **`/synthetic/plot`**  
   - Displays results of fine-tuning on character-based tasks.  
   - The results will match those reported in the paper.

---

### Additional Notes
- The tokenizer has already been trained as part of the dataset is tokenized. However, if needed, you can retrain the tokenizer using the **`tokenizer.train()`** method.  
  **Caution:** Some texts in the dataset include rare characters, such as **Ꞣ**, which are not suitable for small tokenizers. In the original settings, such texts were excluded.  
