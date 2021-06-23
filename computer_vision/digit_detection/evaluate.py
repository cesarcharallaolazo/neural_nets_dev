import gzip
import numpy as np
import random
import os
import pdb

data_dir = 'data/MNIST/raw'
# pdb.set_trace()
with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)
    # print(images)
print(images.shape)
# import matplotlib.pyplot as plt
# image = np.asarray(images[2]).squeeze()
# plt.imshow(image)
# plt.show()

mask = random.sample(range(len(images)), 16) # randomly select some of the test images
mask = np.array(mask, dtype=np.int)
data = images[mask]
print(data.shape)