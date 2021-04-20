'''
Description: 
Author: voicebeer
Date: 2020-12-10 08:47:27
LastEditTime: 2020-12-16 05:14:33
'''

# standard
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
device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")

class DANNet():
    def __init__(self, model=models.DAN(), source_loader=0, target_loader=0, batch_size=64, iteration=10000, lr=0.001, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loader = source_loader
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
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)
        correct = 0

        for i in range(1, self.iteration+1):
            self.model.train()
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
            LEARNING_RATE = self.lr
            # if (i - 1) % 100 == 0:
                # print("Learning rate: ", LEARNING_RATE)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=self.momentum)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
            
            try:
                source_data, source_label = next(source_iter)
            except Exception as err:
                source_iter = iter(self.source_loader)
                source_data, source_label = next(source_iter)
            try:
                target_data, _ = next(target_iter)
            except Exception as err:
                target_iter = iter(self.target_loader)
                target_data, _ = next(target_iter)
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
            
            optimizer.zero_grad()
            source_prediction, mmd_loss = self.model(source_data, data_tgt=target_data)
            cls_loss = F.nll_loss(F.log_softmax(source_prediction, dim=1), source_label.squeeze())
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
            loss = cls_loss + gamma * mmd_loss
            loss.backward()
            optimizer.step()
            # if i % log_interval == 0:
            #     print('Iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}'.format(
            #         i, 100.*i/self.iteration, loss.item(), cls_loss.item(), mmd_loss.item()
            #         )
            #     )
            if i % (log_interval * 20) == 0:
                t_correct = self.test(i)
                if t_correct > correct:
                    correct = t_correct
            #     print('to target max correct: ', correct.item(), "\n")
        return 100. * correct / len(self.target_loader.dataset)

    def test(self, iteration):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in target_loader:
                data = data.to(device)
                target = target.to(device)
                preds, mmd_loss = self.model(data, data)
                test_loss += F.nll_loss(F.log_softmax(preds, dim=1), target.squeeze(), reduction='sum').item()
                pred = preds.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
            test_loss /= len(self.target_loader.dataset)
            # writer.add_scalar("Test/Test loss", test_loss, iteration)

            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(self.target_loader.dataset),
            #     100. * correct / len(self.target_loader.dataset)
            # )) 
        return correct

if __name__ == '__main__':

    dataset_name_all = ['seed3', 'seed4']
    # cross_scenarios = ['cross-session', 'cross-subject']
    # bn_all = ['ele', 'sample', 'global', 'none']
    bn_all = ['none', 'sample']
    
    for dataset_name in dataset_name_all:
        print('Dataset name: ', dataset_name)
        data, label = utils.load_data(dataset_name)
        for bn in bn_all:
            print('Normalization type: ', bn)
            if bn == 'ele':
                data_tmp = copy.deepcopy(data)
                for i in range(len(data_tmp)):
                    for j in range(len(data_tmp[0])):
                        data_tmp[i][j] = utils.norminy(data_tmp[i][j])
            elif bn == 'sample':
                data_tmp = copy.deepcopy(data)
                for i in range(len(data_tmp)):
                    for j in range(len(data_tmp[0])):
                        data_tmp[i][j] = utils.norminx(data_tmp[i][j])
            elif bn == 'global':
                data_tmp = copy.deepcopy(data)
                for i in range(len(data_tmp)):
                    for j in range(len(data_tmp[0])):
                        data_tmp[i][j] = utils.normalization(data_tmp[i][j])
            elif bn == 'none':
                data_tmp = copy.deepcopy(data)
            else:
                pass

            trial_total, category_number, _ = utils.get_number_of_label_n_trial(dataset_name)

            # training settings
            batch_size = 32
            iteration = 15000
            lr = 0.01
            momentum = 0.9
            log_interval = 10

            # meernet = MEERNet(model=models.MEERN())
            
            # store the results
            csub = []
            csesn = []

            # cross-subject, for 3 sessions, 1-14 as sources, 15 as target
            for i in range(3):
                one_session_data, one_session_label = copy.deepcopy(data_tmp[i]), copy.deepcopy(label[i])
                target_data, target_label = one_session_data.pop(), one_session_label.pop()
                source_data, source_label = copy.deepcopy(one_session_data), copy.deepcopy(one_session_label.copy())
                # print(len(source_data))
                source_data_comb = source_data[0]
                source_label_comb = source_label[0]
                for j in range(1, len(source_data)):
                    source_data_comb = np.vstack((source_data_comb, source_data[j]))
                    source_label_comb = np.vstack((source_label_comb, source_label[j]))
                # if bn == 'ele':
                #     source_data_comb = utils.norminy(source_data_comb)
                #     target_data = utils.norminy(target_data)
                # elif bn == 'sample':
                #     source_data_comb = utils.norminx(source_data_comb)
                #     target_data = utils.norminx(target_data)
                # elif bn == 'global':
                #     source_data_comb = utils.normalization(source_data_comb)
                #     target_data = utils.normalization(target_data)
                # elif bn == 'none':
                #     pass
                # else:
                #     pass
                # source_data_comb = utils.norminy(source_data_comb)
                # target_data = utils.norminy(target_data)
                source_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data_comb, source_label_comb),
                                                                        batch_size=batch_size,
                                                                        shuffle=True,
                                                                        drop_last=True)
                # source_loaders = []
                # for j in range(len(source_data)):
                #     source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                #                                                         batch_size=batch_size,
                #                                                         shuffle=True,
                #                                                         drop_last=True))
                target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                                        batch_size=batch_size, 
                                                                        shuffle=True, 
                                                                        drop_last=True)
                model = DANNet(model=models.DAN(pretrained=False, number_of_category=category_number),
                            source_loader=source_loader,
                            target_loader=target_loader,
                            batch_size=batch_size,
                            iteration=iteration,
                            lr=lr,
                            momentum=momentum,
                            log_interval=log_interval)
                # print(model.__getModel__())
                acc = model.train()
                csub.append(acc)
                    
            # cross-session, for 15 subjects, 1-2 as sources, 3 as target
            for i in range(15):
                target_data, target_label = copy.deepcopy(data_tmp[2][i]), copy.deepcopy(label[2][i])
                source_data, source_label = [copy.deepcopy(data_tmp[0][i]), copy.deepcopy(data_tmp[1][i])], [copy.deepcopy(label[0][i]), copy.deepcopy(label[1][i])]
                # one_sub_data, one_sub_label = data[i], label[i]
                # target_data, target_label = one_session_data.pop(), one_session_label.pop()
                # source_data, source_label = one_session_data.copy(), one_session_label.copy()
                # print(len(source_data))
                source_data_comb = np.vstack((source_data[0], source_data[1]))
                source_label_comb = np.vstack((source_label[0], source_label[1]))
                for j in range(1, len(source_data)):
                    source_data_comb = np.vstack((source_data_comb, source_data[j]))
                    source_label_comb = np.vstack((source_label_comb, source_label[j]))
                # if bn == 'ele':
                #     source_data_comb = utils.norminy(source_data_comb)
                #     target_data = utils.norminy(target_data)
                # elif bn == 'sample':
                #     source_data_comb = utils.norminx(source_data_comb)
                #     target_data = utils.norminx(target_data)
                # elif bn == 'global':
                #     source_data_comb = utils.normalization(source_data_comb)
                #     target_data = utils.normalization(target_data)
                # elif bn == 'none':
                #     pass
                # else:
                #     pass
                # source_data_comb = utils.norminy(source_data_comb)
                # target_data = utils.norminy(target_data)

                source_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data_comb, source_label_comb),
                                                                        batch_size=batch_size,
                                                                        shuffle=True,
                                                                        drop_last=True)
                target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                                        batch_size=batch_size, 
                                                                        shuffle=True, 
                                                                        drop_last=True)
                model = DANNet(model=models.DAN(pretrained=False, number_of_category=category_number),
                            source_loader=source_loader,
                            target_loader=target_loader,
                            batch_size=batch_size,
                            iteration=iteration,
                            lr=lr,
                            momentum=momentum,
                            log_interval=log_interval)
                # print(model.__getModel__())
                acc = model.train()
                csesn.append(acc)
            
            # print("Cross-session: ", csesn)
            # print("Cross-subject: ", csub)
            print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
            print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
    