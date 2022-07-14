import numpy as np

folder_prefix = '../samples_256'
filename = 'samples_10000x256x256x3.npz'

a = np.load(f'{folder_prefix}_10000/{filename}')['arr_0']
b = np.load(f'{folder_prefix}_gamma_0_10000/{filename}')['arr_0']

f = np.append(a, b, axis=0)

assert np.all(a == f[:10000])
