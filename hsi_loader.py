import numpy as np
from torch.utils import data
import torch
from PIL import Image
class HSIDataSet(data.Dataset):
    def __init__(self, dataID, setindex='label', max_iters=None, num_unlabel=1000):
        self.setindex = setindex
        if dataID == 1:
            self.root = './dataset/PaviaU/'
        elif dataID == 2:
            self.root = './dataset/Salinas/'
        elif dataID == 3:
            self.root = './dataset/Houston/'
        # elif dataID == 'X031368c':
        #     self.root = './dataset/med1/X031368c/'
        elif dataID == 4:
            self.root = './dataset/Indian_pines/'


        XP = np.load(self.root + 'XP.npy')
        X = np.load(self.root + 'X.npy')
        Y = np.load(self.root + 'Y.npy') - 1

        if self.setindex == 'label':
            train_array = np.load(self.root + 'train_array.npy')
            self.XP = XP[train_array]
            self.X = X[train_array]
            self.Y = Y[train_array]
            if max_iters != None:
                n_repeat = int(max_iters / len(self.Y))
                part_num = max_iters - n_repeat * len(self.Y)
                self.XP = np.concatenate((np.tile(self.XP, (n_repeat, 1, 1, 1)), self.XP[:part_num]))
                self.X = np.concatenate((np.tile(self.X, (n_repeat, 1)), self.X[:part_num]))
                self.Y = np.concatenate((np.tile(self.Y, n_repeat), self.Y[:part_num]))
        elif self.setindex == 'unlabel':
            unlabel_array = np.load(self.root + 'unlabel_array.npy')
            self.XP = XP[unlabel_array[0:num_unlabel]]
            self.X = X[unlabel_array[0:num_unlabel]]
            self.Y = Y[unlabel_array[0:num_unlabel]]
            if max_iters != None:
                n_repeat = int(max_iters / len(self.Y))
                part_num = max_iters - n_repeat * len(self.Y)
                self.XP = np.concatenate((np.tile(self.XP, (n_repeat, 1, 1, 1)), self.XP[:part_num]))
                self.X = np.concatenate((np.tile(self.X, (n_repeat, 1)), self.X[:part_num]))
                self.Y = np.concatenate((np.tile(self.Y, n_repeat), self.Y[:part_num]))
        elif self.setindex == 'test':
            test_array = np.load(self.root + 'test_array.npy')
            self.XP = XP[test_array]
            self.X = X[test_array]
            self.Y = Y[test_array]
        elif self.setindex == 'wholeset':
            self.XP = XP
            self.X = X

    def __len__(self):
        return len(self.X)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5

        if horizontal:
            arrays = np.fliplr(arrays)
        if vertical:
            arrays = np.flipud(arrays)

        return arrays

    @staticmethod
    def Random_rot(arrays):
        # ori = rot_angle <= 0.25
        # rot90 = rot_angle <= 0.5 & rot_angle > 0.25
        # rot180 = rot_angle <= 0.75 & rot_angle > 0.5
        # rot270 = rot_angle <= 1 & rot_angle > 0.75
        rot_angle = np.random.random()
        if rot_angle <= 0.25:
            arrays = arrays
        if rot_angle <= 0.5 and rot_angle > 0.25:
            arrays = np.rot90(arrays)
        if rot_angle <= 0.75 and rot_angle > 0.5:
            arrays = np.rot90(arrays)
            arrays = np.rot90(arrays)
        if rot_angle <= 1 and rot_angle > 0.75:
            arrays = np.rot90(arrays)
            arrays = np.rot90(arrays)
            arrays = np.rot90(arrays)
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __getitem__(self, index):
        # if (self.setindex=='label') or (self.setindex=='test') or (self.setindex=='unlabel'):
        #     XP = self.XP[index].astype('float32')
        #     X = self.X[index].astype('float32')
        #     Y = self.Y[index].astype('int')
        #     return XP.copy(), X, Y

        # --------------------------------------------------------------3-22------------------------------------------------------
        if (self.setindex == 'label') or (self.setindex == 'test') :
            XP = self.XP[index].astype('float32')

            X = self.X[index].astype('float32')
            Y = self.Y[index].astype('int')
            return XP.copy(), X.copy(), Y
        elif self.setindex == 'unlabel':

            XP = self.XP[index].astype('float32')
            X = self.X[index].astype('float32')
            Y = self.Y[index].astype('int')

            return XP.copy(), X.copy(), Y
        else:
            XP = self.XP[index].astype('float32')
            X = self.X[index].astype('float32')
            return XP.copy(), X.copy()
