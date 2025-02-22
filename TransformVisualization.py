import numpy as np
import matplotlib.pyplot as plt
import WaveletTransform
import scipy.io as sio
import WaveletTransform

with open("./dataset/UT_HAR/data/X_train.csv", 'rb') as f:
    data = np.load(f)[0].reshape(90, 250)[0,:].reshape(1, -1)
    data = (data - 42.3199) / 4.9802
plt.plot(data.flatten())
plt.savefig('./data.png')
data_decomp = WaveletTransform.wavelet_decompose(data, q=3, wavelet='db1')
data_decomp = (data_decomp-data_decomp.mean())/data_decomp.std()
plt.plot(data_decomp.flatten())
plt.savefig('./data_decomp.png')
