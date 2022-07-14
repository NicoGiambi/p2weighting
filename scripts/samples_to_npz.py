import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

dir = '../celebahq_p2_samples'
imgs_1 = os.listdir(dir)
img_list = [dir + '/' + imgs_1[i] for i in range(8)]

np_image = np.zeros((len(img_list), 256, 256, 3))

for i, el in enumerate(tqdm(img_list)):
    np_image[i] = Image.open(img_list[i])

print(np_image.shape)
np.savez(f'{dir}/samples_8x256x256x3.npz')
