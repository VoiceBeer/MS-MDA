'''
Description: 
Author: voicebeer
Date: 2020-09-14 01:01:51
LastEditTime: 2021-12-28 01:46:52
'''
# standard
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import random
import time
import math
from torch.utils.tensorboard import SummaryWriter

#
import utils
import models

# random seed


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

# writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MSMDAER():
    def __init__(self, model=models.MSMDAERNet(), source_loaders=0, target_loader=0, batch_size=64, iteration=10000, lr=0.001, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval

    def __getModel__(self):
        return self.model

    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        correct = 0

        for i in range(1, self.iteration+1):
            self.model.train()
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
            LEARNING_RATE = self.lr
            # if (i - 1) % 100 == 0:
                # print("Learning rate: ", LEARNING_RATE)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=self.momentum)
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE)

            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(
                    device), source_label.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()
                cls_loss, mmd_loss, l1_loss = self.model(source_data, number_of_source=len(
                    source_iters), data_tgt=target_data, label_src=source_label, mark=j)
                gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
                beta = gamma/100
                # loss = cls_loss + gamma * (mmd_loss + l1_loss)
                loss = cls_loss + gamma * mmd_loss + beta * l1_loss
                # loss = cls_loss + gamma * (mmd_loss)
                # writer.add_scalar('Loss/training cls loss', cls_loss, i)
                # writer.add_scalar('Loss/training mmd loss', mmd_loss, i)
                # writer.add_scalar('Loss/training l1 loss', l1_loss, i)
                # writer.add_scalar('Loss/training gamma', gamma, i)
                # writer.add_scalar('Loss/training loss', loss, i)
                loss.backward()
                optimizer.step()

                if i % log_interval == 0:
                    print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'.format(
                        i, 100.*i/self.iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()
                    )
                    )
            if i % (log_interval * 20) == 0:
                t_correct = self.test(i)
                if t_correct > correct:
                    correct = t_correct
                # print('to target max correct: ', correct.item(), "\n")
        return 100. * correct / len(self.target_loader.dataset)

    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds)/len(preds)
                test_loss += F.nll_loss(F.log_softmax(pred,
                                        dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.target_loader.dataset)
            # writer.add_scalar("Test/Test loss", test_loss, i)

            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(self.target_loader.dataset),
            #     100. * correct / len(self.target_loader.dataset)
            # ))
            # for n in range(len(corrects)):
            #     print('Source' + str(n) + 'accnum {}'.format(corrects[n]))
        return correct

def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    # one_session_data, one_session_label = copy.deepcopy(data_tmp[session_id]), copy.deepcopy(label[session_id])
    # target_data, target_label = one_session_data.pop(), one_session_label.pop()
    # source_data, source_label = copy.deepcopy(one_session_data[0:source_number]), copy.deepcopy(one_session_label[0:source_number])
    # print("Source number: ", len(source_data))
    
    ## LOSO
    # print(len(data))
    # print(len(data[session_id]))
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    train_idxs = list(range(15))
    del train_idxs[subject_id]
    test_idx = subject_id
    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(one_session_label[train_idxs])
    # print('Target_subject_id: ', test_idx)
    # print('Source_subject_id: ', train_idxs)

    del one_session_label
    del one_session_data

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MSMDAER(model=models.MSMDAERNet(pretrained=False, number_of_source=len(source_loaders), number_of_category=category_number),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    # print(model.__getModel__())
    acc = model.train()
    print('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    return acc

def cross_session(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    # target_data, target_label = copy.deepcopy(data[2][subject_id]), copy.deepcopy(label[2][subject_id])
    # source_data, source_label = [copy.deepcopy(data[0][subject_id]), copy.deepcopy(data[1][subject_id])], [copy.deepcopy(label[0][subject_id]), copy.deepcopy(label[1][subject_id])]

    ## LOSO
    train_idxs = list(range(3))
    del train_idxs[session_id]
    test_idx = session_id
    
    target_data, target_label = copy.deepcopy(data[test_idx][subject_id]), copy.deepcopy(label[test_idx][subject_id])
    source_data, source_label = copy.deepcopy(data[train_idxs][:, subject_id]), copy.deepcopy(label[train_idxs][:, subject_id])

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MSMDAER(model=models.MSMDAERNet(pretrained=False, number_of_source=len(source_loaders), number_of_category=category_number),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    # print(model.__getModel__())
    acc = model.train()
    print('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-MDAER parameters')
    parser.add_argument('--dataset', type=str, default='seed3',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--norm_type', type=str, default='ele',
                        help='the normalization type used for data, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type

    # data preparation
    print('Model name: MS-MDAER. Dataset name: ', dataset_name)
    data, label = utils.load_data(dataset_name)
    print('Normalization type: ', bn)
    if bn == 'ele':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    elif bn == 'sample':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    elif bn == 'global':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    elif bn == 'none':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
    else:
        pass
    trial_total, category_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)

    # training settings
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch*3394/batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch*820/batch_size)
    else:
        iteration = 5000
    print('Iteration: {}'.format(iteration))

    # store the results
    csub = []
    csesn = []
 
    # cross-validation, LOSO
    for session_id_main in range(3):
        for subject_id_main in range(15):
            csub.append(cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
                                    batch_size, iteration, lr, momentum, log_interval))
                                    
    for subject_id_main in range(15):
        for session_id_main in range(3):
            csesn.append(cross_session(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
                                    batch_size, iteration, lr, momentum, log_interval))

    print("Cross-session: ", csesn)
    print("Cross-subject: ", csub)
    print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
    print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
    
