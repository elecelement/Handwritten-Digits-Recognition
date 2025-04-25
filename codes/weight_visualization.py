# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
model.load_model(r'.\saved_models\best_model_2.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

# _, axes = plt.subplots(30, 20)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(600):
#         axes[i].matshow(mats[0].T[i].reshape(28,28))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])

plt.figure()
plt.matshow(mats[1])
plt.xticks([])
plt.yticks([])
plt.show()