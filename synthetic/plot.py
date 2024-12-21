import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys


start = 0
window_size = 8

def moving_average(data, window_size=window_size):


    moving_avg = [sum(data[:window_size])/window_size for i in range(window_size)]
    #sys.exit()
    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        window_mean = np.mean(window)
        moving_avg.append(window_mean)
    return moving_avg

while True:
    # Load the data from the pickle file
    with open('training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)
        counter = 0
        step_i = 0
        step_old = -100
        step_shift = 0
        new_steps = []
        new_training = 0
        for step in training_data['step']:
            add_step = step+step_shift
            new_steps.append(add_step)
            if step<step_old:
                step_shift = step_old - 1800
                new_steps[-1] = new_steps[-1]+step_shift
                new_training = new_steps[-1]
            #print(add_step)
            step_old = step
            counter +=1
            #print(new_steps[-1])
            #if counter == 10:
            #    break
        #print(training_data)
        #training_data['step'] = 
    # Extract data for plotting
    steps = new_steps#training_data['step']
    #print(steps[-5:])
    #print('new_steps:',)

    # Extract losses for each category
    generic_train_original = training_data['loss']['train']['generic']['original']
    generic_train_chars = training_data['loss']['train']['generic']['chars']
    generic_val_original = training_data['loss']['val']['generic']['original']
    generic_val_chars = training_data['loss']['val']['generic']['chars']

    synthetic_train_original = training_data['loss']['train']['synthetic']['original']
    synthetic_train_chars = training_data['loss']['train']['synthetic']['chars']
    synthetic_val_original = training_data['loss']['val']['synthetic']['original']
    synthetic_val_chars = training_data['loss']['val']['synthetic']['chars']
    #print('steps:',steps[0],steps[-1],len(steps))
    #print('generic_val_original:',generic_val_original[0],generic_val_original[-1],len(generic_val_original))
    #print('generic_val_original_mv:',generic_val_original[0],generic_val_original[-1],len(moving_average(generic_val_original)))
    #a = moving_average(generic_val_original, 8)
    # Plot for generic data
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    #print('len:',len(steps[start::8]))
    #print('len:',len(generic_val_original[start::8]))
    #print('len:',len(moving_average(generic_val_original, window_size, 6600)))
    #plt.plot(steps[start:], generic_train_original[start:], color='lightblue', label='Train Original (Generic)')
    #plt.plot(steps[start:], generic_train_chars[start:], color='pink', label='Train Chars (Generic)')
    plt.plot(steps[start:], moving_average(generic_val_original)[start:], color='blue', label='Val Original (LM)')
    plt.plot(steps[start:], moving_average(generic_val_chars)[start:], color='red', label='Val Chars (LM)')
    #plt.axvline(x=new_training , color='green', linestyle='--', label='')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Language modeling cross-entropy loss')
    plt.legend()
    plt.grid(True)

    # Plot for synthetic data
    plt.subplot(1, 2, 2)
    #plt.plot(steps[start:], synthetic_train_original[start:], color='lightblue', label='Train Original (Synthetic)')
    #plt.plot(steps[start:], synthetic_train_chars[start:], color='pink', label='Train Chars (Synthetic)')
    plt.plot(steps[start:], moving_average(synthetic_val_original)[start:], color='blue', label='Val Original (Synthetic)')
    plt.plot(steps[start:], moving_average(synthetic_val_chars)[start:], color='red', label='Val Chars (Synthetic)')
    #plt.axvline(x=new_training , color='green', linestyle='--', label='') 
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Synthetic character-level tasks cross-entropy loss')
    plt.legend()
    plt.grid(True)

    # Display the plots
    plt.tight_layout()
    plt.show()
