# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import rotate, shift, zoom
import numpy as np

def rotate_image(image, angle_range=(-15, 15)):
    angle = np.random.uniform(*angle_range)
    rotated_image = rotate(image.reshape(28, 28), angle, reshape=False)
    return np.resize(rotated_image, (28, 28))  # 确保大小为(28, 28)

def shift_image(image, shift_range=(-3, 3)):
    shift_x = np.random.randint(*shift_range)
    shift_y = np.random.randint(*shift_range)
    shifted_image = shift(image.reshape(28, 28), (shift_x, shift_y), mode='nearest')
    return np.resize(shifted_image, (28, 28))  # 确保大小为(28, 28)

def zoom_image(image, zoom_range=(0.9, 1.1)):
    zoom_factor = np.random.uniform(*zoom_range)
    zoomed_image = zoom(image.reshape(28, 28), zoom_factor, order=1)
    return np.resize(zoomed_image, (28, 28))  # 确保大小为(28, 28)

def flip_image(image):
    if np.random.rand() > 0.5:
        flipped_image = np.fliplr(image.reshape(28, 28))
    else:
        flipped_image = image.reshape(28, 28)
    return flipped_image  # 保证大小为(28, 28)

def augment_images(images):
    augmented_images = []
    for image in images:
        augmented_image = image
        augmented_image = rotate_image(augmented_image)
        augmented_image = shift_image(augmented_image)
        augmented_image = zoom_image(augmented_image)
        augmented_image = flip_image(augmented_image)
        augmented_images.append(augmented_image.flatten())  # 变换后仍然展平为1D
    return np.array(augmented_images)

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]
augmented_train_imgs = augment_images(train_imgs)
augmented_train_labs = np.tile(train_labs[:len(augmented_train_imgs)], (len(augmented_train_imgs)//len(train_labs), 1)).flatten()

# Combine the augmented data with the original data
train_imgs_combined = np.concatenate([train_imgs, augmented_train_imgs])
train_labs_combined = np.concatenate([train_labs, augmented_train_labs])
# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()
train_imgs_combined = train_imgs_combined / train_imgs_combined.max()

linear_model = nn.models.Model_MLP([train_imgs_combined.shape[-1], 256,10], 'ReLU', [1e-4, 1e-4])
#linear_model = nn.models.Model_CNN([1, 32, 64],[256, 128, 10],[32,1,28,28], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
#optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=linear_model,beta = 0.9)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
#scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.999)
#scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer,step_size=200,gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

# runner for MLP
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
# runner for CNN
#runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()