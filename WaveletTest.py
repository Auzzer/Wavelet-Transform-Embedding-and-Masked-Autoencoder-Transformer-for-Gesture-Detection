import numpy as np
import pywt
import torch
import load_data
from torch.utils.data import Dataset, DataLoader

def pad(length, q):
    divisor = 2 ** q
    pad_number = 0
    if length % divisor == 0:
        return pad_number

    # Calculate the smallest x that makes (n + x) divisible by 2^m
    remainder = length % divisor
    pad_number = divisor - remainder

    return pad_number


def wavelet_decompose(unzip_dataloader, q=3, wavelet='db1'):
    # In case the length cannot be divided by the 2^q, add some empty values to the data
    series_length = unzip_dataloader[0].shape[-1]
    pad_number = pad(series_length, q)
    X_decompsed = []
    for batch in unzip_dataloader:
        batch_decomposed = []
        for data in batch:
            data_decomposed = []
            data = np.pad(data, ((0, 0), (0, pad_number)), mode='constant', constant_values=0)

            for channel in data:
                coeffs = pywt.wavedec(channel, wavelet, level=q)
                embedding_shape = coeffs[0].shape[0]
                data_decomposed.append(np.concatenate(coeffs))
            batch_decomposed.append(np.array(data_decomposed))
        X_decompsed.append(np.array(batch_decomposed))

    return embedding_shape, X_decompsed

class wavelet_transform:
    def __init__(self, q=3, wavelet = 'db1'):
        self.q = q
        self.wavelet = wavelet

    def __call__(self, sample):
        inputs, targets = sample
        q = self.q
        wavelet = self.wavelet
        _, inputs = wavelet_decompose(inputs, q=q, wavelet=wavelet)
        return inputs, targets


trainloader, testloader = load_data.load_data('./dataset/', 'Widar')
X_train, y_train = zip(*[(data.numpy(), label) for data, label in trainloader])
X_test, y_test = zip(*[(data.numpy(), label) for data, label in testloader])

embedding_shape, X_train_decompose = wavelet_decompose(X_train, q=3, wavelet='db1')
_, X_test_decompose = wavelet_decompose(X_test, q=3, wavelet='db1')

X_train_decompose_flatten = [item for sublist in X_train_decompose for item in sublist]
X_test_decompose_flatten = [item for sublist in X_test_decompose for item in sublist]

y_train_flatten = [item for sublist in y_train for item in sublist]
y_test_flatten = [item for sublist in y_test for item in sublist]

class DecomposedDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels[idx]
        data_tensor = torch.tensor(data, dtype=torch.float32)

        return data_tensor, label

y_train_flatten = [item for sublist in y_train for item in sublist]
y_test_flatten = [item for sublist in y_test for item in sublist]

trainloader_decomposed  =  DataLoader(DecomposedDataset(X_train_decompose_flatten, y_train_flatten),
                                      batch_size=64, shuffle=True, num_workers=12)

testloader_decomposed  =  DataLoader(DecomposedDataset(X_test_decompose_flatten, y_test_flatten),
                                      batch_size=64, shuffle=False, num_workers=12)


#For testing
for data, labels in trainloader_decomposed:
    print("Data shape:", data.shape)   # should be [64, 22, 400]
    print("Labels shape:", labels.shape)

