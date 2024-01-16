from tools.hyper_tools import *
import argparse
import os

def featureNormalize(X, type):
    # type==1: x = (x-mean)/std(x)
    # type==2: x = (x-max(x))/(max(x)-min(x))
    if type == 1:
        mu = np.mean(X, 0)
        X_norm = X - mu
        sigma = np.std(X_norm, 0)
        X_norm = X_norm / sigma
        return X_norm
    elif type == 2:
        minX = np.min(X, 0)
        maxX = np.max(X, 0)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX)
        return X_norm

def main(args):
    dataID = args.dataID
    n_PC = args.n_PC
    w = args.w
    num_label = args.num_label
    XP,X,Y = SampleGen(dataID=dataID,w=w,n_PC=n_PC)

    if dataID==1:
        save_pre_dir = './dataset/PaviaU/'
    elif dataID==2:
        save_pre_dir = './dataset/Salinas/'
    elif dataID==3:
        save_pre_dir = './dataset/Houston/'
    elif dataID==4:
        save_pre_dir = './dataset/Indian_pines/'

    if os.path.exists(save_pre_dir)==False:
        os.makedirs(save_pre_dir)
    # XP: m*n_PC*w*w
    # X: m*n_band
    # Y: m*1

    n_class = Y.max()

    train_num_array = np.ones((n_class,)).astype('int')*num_label

    # np.random.seed(2) 94.36
    np.random.seed(2)
    randomArray_wholeset = np.where(Y>0)[0]
    np.random.shuffle(randomArray_wholeset)

    for i in range(1,n_class+1):
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        # np.random.seed(0)
        np.random.seed(0)
        randomArray_label = np.random.permutation(n_data)
        if i==1:
            train_array = index[randomArray_label[0:train_num_array[i-1]]]
            test_array = index[randomArray_label[train_num_array[i-1]:n_data]]
        else:
            train_array = np.append(train_array,index[randomArray_label[0:train_num_array[i-1]]])
            test_array = np.append(test_array,index[randomArray_label[train_num_array[i-1]:n_data]])

    unlabel_array = np.array(list(set(randomArray_wholeset)-set(train_array)))
    # unlabel_array = unlabel_array[:10000]

    np.save(save_pre_dir+'XP.npy',XP)
    np.save(save_pre_dir+'X.npy',X)
    np.save(save_pre_dir+'Y.npy',Y)
    np.save(save_pre_dir+'train_array.npy',train_array)
    np.save(save_pre_dir+'test_array.npy',test_array)
    np.save(save_pre_dir+'unlabel_array.npy',unlabel_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--num_label', type=int, default=5)
    parser.add_argument('--w', type=int, default=20)
    parser.add_argument('--n_PC', type=int, default=60)

    main(parser.parse_args())