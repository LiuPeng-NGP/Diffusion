import torch
from torchvision.datasets import CIFAR10
import os

train_set = CIFAR10("./data", train=True, download=True)
print("CIFAR10 train dataset:", len(train_set))

images = []
labels = []
for img, label in train_set:
    images.append(img)
    labels.append(label)

labels = torch.tensor(labels)
for i in range(10):
    assert (labels == i).sum() == 5000

output_dir = "./data/cifar10-pngs/"
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
for i, pil in enumerate(images):
    pil.save(os.path.join(output_dir, "{:05d}.png".format(i)))