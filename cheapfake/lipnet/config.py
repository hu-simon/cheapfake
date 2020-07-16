# Basic options.
gpu = "0"
random_seed = 0
mode = "unseen"
video_path = "/Users/shu/Documents/Datasets/LIPS/GRID_LIP"
train_list = "/Users/shu/Documents/Datasets/LIPS/data/{}_train.txt".format(mode)
val_list = "/Users/shu/Documents/Datasets/LIPS/data/{}_val.txt".format(mode)
annotations_path = "/Users/shu/Documents/Datasets/LIPS/GRID_align_txt"

# Video and text options.
video_padding = 75
text_padding = 200

# Training options.
batch_size = 96
base_learning_rate = 2e-5
num_workers = 4
max_epochs = 10000
optimize = True
train_write_rate = 1
validation_write_rate = 1000
test_write_rate = 1000

# Save options.
save_prefix = "weights/LipNet_{}".format(mode)

# Weights.
weights = "/Users/shu/Documents/Datasets/LIPS/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt"
