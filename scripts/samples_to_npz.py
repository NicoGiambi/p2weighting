import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

dir1 = 'samples_256_10000_2'
dir2 = 'samples_256_10000_3'
imgs_1 = os.listdir(dir1)
imgs_2 = os.listdir(dir2)
img_list = [dir1 + '/' + imgs_1[i] for i in range(5000)]
img_list += [dir2 + '/' + imgs_2[i] for i in range(5000)]

np_image = np.zeros((len(img_list), 256, 256, 3))

for i, el in enumerate(tqdm(img_list)):
    np_image[i] = Image.open(img_list[i])

print(np_image.shape)
os.mkdir('samples_256_10000_2_3')
np.savez('samples_256_10000_2_3/samples_10000x256x256x3.npz')
