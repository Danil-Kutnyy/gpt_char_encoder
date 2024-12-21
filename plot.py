import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load training data
with open("training_data.pkl", "rb") as f:
    training_data = pickle.load(f)

def moving_average(data, window_size, start=0):

    if window_size <= 1:
        return data
    else:
        moving_avg = [sum(data[:window_size])/window_size for i in range(window_size)]
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            window_mean = np.mean(window)
            moving_avg.append(window_mean)
        return moving_avg[ int(start/window_size): ]


# Define the range for plotting
start = 32
end = -1
mv = 16




# Delta Calculation
delta_val = [
    val_orig - val_char
    for val_char, val_orig in zip(training_data["val_loss_char_encoder_add"][start:end],
                                  training_data["val_loss_original"][start:end])
]
delta_train = [
    val_orig - val_char
    for val_char, val_orig in zip(training_data["train_loss_char_encoder_add"][start:end],
                                  training_data["train_loss_original"][start:end])
]


#mooving avarage
training_data["val_loss_original"] = new_data = moving_average(training_data["val_loss_original"], mv, start)
training_data["train_loss_original"] = new_data = moving_average(training_data["train_loss_original"], mv, start)
training_data["val_loss_char_encoder_add"] = new_data = moving_average(training_data["val_loss_char_encoder_add"], mv, start)
training_data["train_loss_char_encoder_add"] = new_data = moving_average(training_data["train_loss_char_encoder_add"], mv, start)
delta_val = new_data = moving_average(delta_val, mv, start)
delta_train = new_data = moving_average(delta_train, mv, start)



# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Plot original and Char Encoder Add losses in the first subplot
ax1.plot(training_data["val_loss_original"][start:end], label="Original Model - Validation Loss", color="blue")
ax1.plot(training_data["train_loss_original"][start:end], label="Original Model - Training Loss", color="lightblue")
ax1.plot(training_data["val_loss_char_encoder_add"][start:end], label="Char Encode Model - Validation Loss", color="red")
ax1.plot(training_data["train_loss_char_encoder_add"][start:end], label="Char Encode Model - Training Loss", color="pink")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.set_title("Training Progress")

# Plot delta in the second subplot
ax2.plot(delta_val[16:], label="Delta - Validation", color="darkgreen", linestyle="--")
ax2.plot(delta_train[16:], label="Delta - Training", color="lightgreen", linestyle="--")
ax2.axhline(0, color="black", linestyle=":")  # Baseline at y=0
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Delta (Loss)")
ax2.legend()
ax2.set_title("Delta | Char Encode and Original | Validation Loss")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

