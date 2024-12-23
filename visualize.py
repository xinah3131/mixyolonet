import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df_resnet18 = pd.read_csv(r'C:\Users\Lenovo\Desktop\MMU\FYP\Code\Yolonet\weights\56_640_0.0001_0.01_0.1_0.9_3dec_aug_msb(56).csv')

# Extract necessary columns
epochs = df_resnet18['epoch']
detect_loss = df_resnet18['detection_loss']
eval_detect_loss = df_resnet18['valid_detection_loss']
dehazing_loss = df_resnet18['dehazing_loss']
valid_dehazing_loss = df_resnet18['valid_dehazing_loss']
map50 = df_resnet18['mAP@50']

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Plot detecting loss and eval detecting loss
axes[0].plot(epochs, detect_loss, label='Detecting Loss', color='blue')
axes[0].plot(epochs, eval_detect_loss, label='Eval Detecting Loss', color='orange')
axes[0].set_title('Detecting Loss vs Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Plot dehazing loss and eval dehazing loss
axes[1].plot(epochs, dehazing_loss, label='Dehazing Loss', color='green')
axes[1].plot(epochs, valid_dehazing_loss, label='Eval Dehazing Loss', color='red')
axes[1].set_title('Dehazing Loss vs Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

# Plot mAP@50
axes[2].plot(epochs, map50, label='mAP@50', color='purple')
axes[2].set_title('mAP@50 vs Epochs')
axes[2].set_xlabel('Epochs')
axes[2].set_ylabel('mAP@50')
axes[2].legend()
axes[2].grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('output_plot.png')

# # Show the plot
# plt.show()