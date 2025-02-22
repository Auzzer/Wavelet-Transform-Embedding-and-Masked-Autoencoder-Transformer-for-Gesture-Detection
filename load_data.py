import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import WaveletTransform

# 1. Read the data. Wavelet Transform is included in this step/

def UT_HAR_dataset(root_dir, q=2, wavelet = 'db1'):
    # [90, 250] in [channel, length]
    data_list = glob.glob(root_dir + '/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir + '/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 90, 250)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            data_decomposed = []
            for sample in data_norm:
                sample = WaveletTransform.wavelet_decompose(sample, q=q, wavelet=wavelet)
                data_decomposed.append(sample)
            data_norm = np.array(data_decomposed)

        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data # data['X_train']
# embedding size = data.shape[-1]/(2**q-1)
#
class CSI_Dataset(Dataset):
    """CSI dataset.
    NTU-Fi-HumanID and NTU-Fi_HAR all use this one.
    Input: dataset: /class_name/xx.mat. For example, './dataset/NTU-Fi_HAR/tain_amp'
    Output: torch dataset class in the form of (tensor([....]), label)
    """

    def __init__(self, root_dir, q=2, wavelet='db1',modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal = modal
        self.q = q
        self.wavelet = wavelet
        self.transform = transform
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal] # [342, 2000]

        # normalize
        x = (x - 42.3199) / 4.9802
        x = WaveletTransform.wavelet_decompose(x, q = self.q, wavelet=self.wavelet)
        """
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        """
        if self.transform:
            x = self.transform(x)

        x = torch.FloatTensor(x)

        return x, y


class Widar_Dataset(Dataset):
    """
    input: root + 'Widardata/train/'
    output: same as CSI_dataset

    """
    def __init__(self, root_dir, wavelet, q):
        self.root_dir = root_dir
        self.q = q
        self.wavelet = wavelet
        self.data_list = glob.glob(root_dir + '/*/*.csv')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')

        # normalize
        x = (x - 0.0025) / 0.0119

        # x = WaveletTransform.wavelet_decompose(x, q=self.q, wavelet=self.wavelet)
        # need to be rewritten for the 2D dimensional

        # reshape: (22,400) -> 22,20,20
        x = x.reshape(22, 20, 20)
        # interpolate from 20x20 to 32x32
        x = self.reshape(x)
        if self.transform:
            x = self.transform(x)
            

        x = torch.FloatTensor(x)

        return x, y



# Create the dataloader
def load_data(root, dataset_name, q=2, wavelet='db1', batch_size=64, num_workers=16):
    classes = {'UT_HAR_data': 7, 'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6, 'Widar': 22}
    ##########################################################################
    if dataset_name == 'UT_HAR_data':
        num_classes = classes['UT_HAR_data']
        print('UT-HAR shape: [90, 250] in [channel, length]')
        data = UT_HAR_dataset(root, q=q, wavelet=wavelet)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']

        train_set = torch.utils.data.TensorDataset(X_train, data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((X_val, X_test), 0),
                                                  torch.cat((data['y_val'], data['y_test']), 0))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,
                                                   drop_last=True, num_workers=num_workers)  # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers)


    ##########################################################################
    elif dataset_name == 'NTU-Fi-HumanID':
        print('NTU-Fi-HumanID shape: [342, 2000] in [channel, length]')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/', q=q, wavelet=wavelet),
                                                   batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/', q=q, wavelet=wavelet),
                                                  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ##########################################################################
    elif dataset_name == 'NTU-Fi_HAR':
        print('NTU-Fi_HAR shape: [342, 2000] in [channel, length]')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/', q=q, wavelet=wavelet),
                                                   batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/', q=q, wavelet=wavelet),
                                                  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ##########################################################################
    elif dataset_name == 'Widar':
        print('Widar shape: [22, 400] in [channel, length]')
        num_classes = classes['Widar']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/', wavelet=wavelet, q=q),
                                                   batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/', wavelet=wavelet, q=q),
                                                  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes
"""
for batch in train_loader:
    series_len=batch[0][1]
    break
"""