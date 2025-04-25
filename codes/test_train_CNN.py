# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle


np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:300]
valid_labs = train_labs[:300]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 128, 10], 'ReLU', [1e-4, 1e-4])
CNN_model = nn.models.Model_CNN(conv_size_list=[1,8],
    linear_size_list=[1352,128,10],#原来为128
    image_info=[32, 1, 28, 28],
    act_func='ReLU',
    lambda_list=[1e-4, 1e-4, 1e-4, 1e-4],
    kernel_size=3)
optimizer = nn.optimizer.SGD(init_lr=0.06, model=CNN_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100,200,250,300,350,400,450,500,550], gamma=0.5)
#scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
loss_fn = nn.op.MultiCrossEntropyLoss(model=CNN_model, max_classes=train_labs.max() + 1)
runner = nn.runner.RunnerM(CNN_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler,is_CNN = True)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=10, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()