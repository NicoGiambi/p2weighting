import numpy as np

folder_prefix = 'samples_256_10000_'
filename = 'samples_10000x256x256x3.npz'

a = np.load(f'{folder_prefix}0/{filename}')['arr_0']
b = np.load(f'{folder_prefix}1/{filename}')['arr_0']

f = np.append(a, b, axis=0)
print('a, b')
del a
del b

c = np.load(f'{folder_prefix}2/{filename}')['arr_0']
d = np.load(f'{folder_prefix}3/{filename}')['arr_0']

g = np.append(c, d, axis=0)
print('c, d')
del c
del d

h = np.append(f, g, axis=0)
print('a, b, c, d')
del f
del g

e = np.load(f'{folder_prefix}4/{filename}')['arr_0']
i = np.append(h, e, axis=0)
print('a, b, c, d, e')
del e

np.savez('samples_50000x256x256x3.npz', i)
print(i.shape)
