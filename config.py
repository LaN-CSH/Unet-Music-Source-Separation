
# import numpy as np
window_size = 2048
hop_length = 1536

patch_size = 256  # roughly 9 seconds
sample_size = 393728  # (255*1536 + 2048)

# 44100/sec
# for training
EPOCH = 100
BATCH = 217
test_BATCH = 203

# print(np.load(r'E:\Stem_Np\Test\PR - Oh No.stem.mp4_np.npy').shape)

