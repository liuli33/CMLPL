import torch
from tools.hyper_tools import *
import argparse
import time
from tools.models import *
from torch.nn import functional as F
from hsi_loader import HSIDataSet
from torch.utils import data
import os
import random
from regularizer import Distribution_Loss
import os
# torch.set_printoptions(profile="full")
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
import pandas as pd
DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Houston', 8001: '8001'}
LABELED_FEAT_TABLES = None
UNLABELED_FEAT_TABLES = None
print(222222222222)
def __init_weight(feature, conv_init,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_weight(module_list, conv_init,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init,
                      **kwargs)



def label_onehot(inputs, num_segments):
    inputs = inputs.unsqueeze(1)
    batch_size = inputs.shape[0]
    one_hot = torch.zeros(batch_size, num_segments).cuda()
    one_hot = one_hot.scatter_(1, inputs.cuda(), 1)

    return one_hot


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# seed_torch(seed=1088)
def seed_torch(seed=1088):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

@torch.no_grad()
def _update_bank(self, k, labels, index):
    self.bank[:, index] = k.t()
    self.labels[index] = labels
def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets+epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5

def main(args):

    kappa = []
    aa = []
    oa = []
    all_acc = []

    kappa1 = []
    aa1 = []
    oa1 = []
    all_acc1 = []


    if args.dataID == 1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './dataset/PaviaU/'
    elif args.dataID == 2:
        num_classes = 16
        num_features = 204
        save_pre_dir = './dataset/Salinas/'
    elif args.dataID == 3:
        num_classes = 15
        num_features = 144
        save_pre_dir = './dataset/Houston/'
    elif args.dataID == 4:
        num_classes = 16
        num_features = 200
        save_pre_dir = './dataset/Indian_pines/'
    Y = np.load(save_pre_dir + 'Y.npy') - 1
    test_array = np.load(save_pre_dir + 'test_array.npy')
    Y = Y[test_array]
    print_per_batches = args.print_per_batches

    save_path_prefix = args.save_path_prefix + 'Experiment_' + repr(args.dataID) + \
                       '/label_' + repr(args.num_label) + '/'

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)

    LABELED_FEAT_TABLES = None
    UNLABELED_FEAT_TABLES = None
    labeled_loader = data.DataLoader(
        HSIDataSet(args.dataID, setindex='label', max_iters=args.num_unlabel),
        batch_size=args.labeled_batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker,
        pin_memory=False)

    unlabeled_loader = data.DataLoader(
        HSIDataSet(args.dataID, setindex='unlabel', max_iters=args.num_unlabel, num_unlabel=args.num_unlabel),
        batch_size=args.unlabeled_batch_size, shuffle=True, num_workers=args.num_workers,
        worker_init_fn=seed_worker, pin_memory=False)

    whole_loader = data.DataLoader(
        HSIDataSet(args.dataID, setindex='wholeset', max_iters=None),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker,
        pin_memory=False)

    for index_iter in range(1):
        Base = CCT_Net(num_features=num_features,
                        dropout=args.dropout,
                        num_classes=num_classes,
                        )

        Decoder_base = classifier(num_classes)
        Decoder1 = classifier(num_classes)
        Decoder2 = classifier(num_classes)



        Base = Base.cuda()
        Decoder_base = Decoder_base.cuda()
        Decoder1 = Decoder1.cuda()
        Decoder2 = Decoder2.cuda()

        cls_loss = torch.nn.CrossEntropyLoss()
        base_optimizer = torch.optim.Adam(list(Base.parameters())+list(Decoder_base.parameters()), lr=args.lr)
        aug_decoder_optimizer = torch.optim.Adam(list(Base.parameters())+list(Decoder1.parameters())+list(Decoder2.parameters()), lr=args.lr)
        num_batches = min(len(labeled_loader), len(unlabeled_loader))

        num_steps = args.num_epochs * num_batches
        loss_hist = np.zeros((num_steps, 5))
        index_i = -1
        for epoch in range(args.num_epochs):

            for batch_index, (labeled_data, unlabeled_data) in enumerate(zip(labeled_loader, unlabeled_loader)):
                index_i += 1
                base_optimizer.zero_grad()
                aug_decoder_optimizer.zero_grad()

                # train with labeled data
                Base.train()
                Decoder_base.train()
                XP_train, X_train, Y_train = labeled_data
                XP_train1 = XP_train.cuda() + torch.randn(XP_train.size()).cuda() * args.noise
                X_train1 = X_train.cuda() + torch.randn(X_train.size()).cuda() * args.noise
                Y_train = Y_train.cuda()
                batch_size = XP_train.size(0)
                # labeled_output, x_feature = Base(XP_train1, X_train1)

                Decoder1.train()
                Decoder2.train()
                # train with unlabeled data
                XP_un, X_un, _ = unlabeled_data
                XP_un = XP_un.cuda()
                X_un = X_un.cuda()

                # loss_labeled

                XP_b_input = XP_un + torch.randn(XP_un.size()).cuda() * args.noise  # seem like strong aug
                X_b_input = X_un + torch.randn(X_un.size()).cuda() * args.noise
                XP_b_all = torch.cat([XP_train1, XP_b_input], dim=0)
                X_b_all = torch.cat([X_train1, X_b_input], dim=0)
                un_b_output_all, xs_feature_all = Base(XP_b_all, X_b_all)

                x_feature = xs_feature_all[:batch_size, :]
                labeled_output = Decoder_base(x_feature)
                cls_loss_value = cls_loss(labeled_output, Y_train)
                # unlable
                un_feature = un_b_output_all[batch_size:, :]
                un_feature_AUG1 = un_feature + torch.randn(un_feature.size()).cuda() * args.noise
                un_feature_AUG2 = un_feature + torch.randn(un_feature.size()).cuda() * args.noise

                origin_out = Decoder_base(un_feature)
                aug_out1 = Decoder1(un_feature_AUG1)
                aug_out2 = Decoder2(un_feature_AUG2)
                ori_targets =  F.softmax(origin_out.detach(), dim=1)
                targets1 = F.softmax(aug_out1.detach(), dim=1)
                targets2 = F.softmax(aug_out2.detach(), dim=1)

                all_loss = cls_loss_value + softmax_js_loss(origin_out, targets1) + softmax_js_loss(origin_out, targets2) + softmax_js_loss(aug_out1, ori_targets) + softmax_js_loss(aug_out2, ori_targets)
                _, labeled_prd_label = torch.max(labeled_output, 1)

                # ---------------------------------------------------------------3-29-尾部-----------------------------------------------------------------------------------------
                # cls_loss_value 优化 b, con_loss_value 优化b, cossis 优化 b

                all_loss.backward()
                base_optimizer.step()
                aug_decoder_optimizer.step()


                # update base and ensemble networks

                # training stat
                loss_hist[index_i, 0] = cls_loss_value.item()
                loss_hist[index_i, 1] = cls_loss_value.item()
                loss_hist[index_i, 2] = cls_loss_value.item()
                loss_hist[index_i, 3] = cls_loss_value.item()
                loss_hist[index_i, 4] = torch.mean((labeled_prd_label == Y_train).float()).item()  # acc
                tem_time = time.time()

                if (batch_index + 1) % print_per_batches == 0:
                    print(
                        'Epoch %d/%d:  %d/%d loss_contrast= %.2f total_loss = %.4f cls_loss = %.4f con_loss = %.4f acc = %.2f\n' \
                        % (epoch + 1, args.num_epochs, batch_index + 1, num_batches,
                           np.mean(loss_hist[index_i - print_per_batches + 1:index_i + 1, 0]),
                           np.mean(loss_hist[index_i - print_per_batches + 1:index_i + 1, 1]),
                           np.mean(loss_hist[index_i - print_per_batches + 1:index_i + 1, 2]),
                           np.mean(loss_hist[index_i - print_per_batches + 1:index_i + 1, 3]),
                           np.mean(loss_hist[index_i - print_per_batches + 1:index_i + 1, 4]) * 100))
        time1 = time.time()
        predict_label = CCT_test_whole(Base, Decoder_base, whole_loader, print_per_batches=10)
        time2 = time.time()
        print('推理时间为==', time2 - time1)

        predict_test = predict_label[test_array]
        OA, Kappa, producerA = CalAccuracy(predict_test, Y)
        print('Result:\n OA=%.2f,Kappa=%.2f' % (OA * 100, Kappa * 100))
        print('producerA:', producerA * 100)
        print('AA=%.2f' % (np.mean(producerA) * 100))

        img = DrawResult(predict_label + 1, args.dataID)
        # plt.imsave(save_path_prefix + 'our*4_con_2.5ctr_0.95h' + 'OA_' + repr(int(OA * 10000)) + '.svg', img, transparent = True)
        plt.imsave(save_path_prefix + 'IP_NEW_CCT' + 'OA_' + repr(int(OA * 10000)) + '.svg', img, dpi=300)
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去刻度
        plt.imshow(img)
        # plt.savefig(save_path_prefix + 'IP_OUR_NEW_NEG' + 'OA_' + repr(int(OA * 10000)) + '.svg',
        #            dpi=1000, bbox_inches='tight',transparent = True, pad_inches = 0)

        kappa.append(Kappa)
        oa.append(OA)
        aa.append(np.mean(producerA))
        all_acc.append(producerA)


    OA = np.mean(oa)*100
    OA_std = np.std(oa)*100
    AA = np.mean(aa)*100
    AA_std = np.std(aa)*100
    KAPPA = np.mean(kappa)*100
    KAPPA_std = np.std(kappa)*100
    ALL_ACC = np.mean(all_acc, axis=0)*100
    ALL_ACC_std = np.std(all_acc, axis=0)*100


    dataframe = pd.DataFrame({'OA': OA, 'OA_std': OA_std, 'AA': AA, 'AA_std': AA_std,
                              'KAPPA': KAPPA, 'KAPPA_std': KAPPA_std, 'ALL_ACC': ALL_ACC, 'ALL_ACC_std': ALL_ACC_std
                              })
    dataframe.to_csv("OUR_HU_respnoe""+.csv", index=False, sep=',')
    print('mean_OA ± std_OA is: ' + str(np.mean(oa)) + ' ± ' + str(np.std(oa)) + '\n')
    print('mean_AA ± std_AA is: ' + str(np.mean(aa)) + ' ± ' + str(np.std(aa)) + '\n')
    print('mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa)) + ' ± ' + str(np.std(kappa)) + '\n' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=str, default=1)
    parser.add_argument('--num_label', type=int, default=5)

    parser.add_argument('--save_path_prefix', type=str, default='./')

    # train
    parser.add_argument('--labeled_batch_size', type=int, default=128)
    parser.add_argument('--unlabeled_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--print_per_batches', type=int, default=10)

    parser.add_argument('--num_unlabel', type=int, default=10000)

    # -------------------------------------3-29-add--------------------------
    parser.add_argument('--thr', type=float, default=1,
                        help='pseudo label threshold')
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--queue-batch', type=float, default=17,
                        help='number of batches stored in memory bank')
    parser.add_argument('--temperature', default=0.3, type=float, help='softmax temperature')

    # network
    parser.add_argument('--teacher_alpha', type=float, default=0.95)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--noise', type=float, default=0.5)
    parser.add_argument('--m', type=int, default=5, help='number of stochastic augmentations')

    main(parser.parse_args())
