#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar
from mixup import mixup_data, mixup_criterion
from resnet import *

class myModel(nn.Module):
    
    def __init__(self,name):
        super(myModel, self).__init__()
#         parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
        parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
        parser.add_argument('--name', default='0', type=str, help='name of run')
        parser.add_argument('--seed', default=0, type=int, help='random seed')
        parser.add_argument('--batch-size', default=128, type=int, help='batch size')
        parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
        parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
        parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
        parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
#         args = parser.parse_args()
        args, unknown = parser.parse_known_args()
        self.arg=args
        
        self.use_cuda = torch.cuda.is_available()

        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        if args.seed != 0:
            torch.manual_seed(args.seed)

# Data

# Model
        if args.resume:
    # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
            self.net = checkpoint['net']
            self.best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            rng_state = checkpoint['rng_state']
            torch.set_rng_state(rng_state)
        else:
            print('==> Building model..')
            self.net = models.__dict__[name]()
#             self.net=ResNet18()

        if not os.path.isdir('results'):
            os.mkdir('results')
        self.logname = ('results/log_' + self.net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

        if self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            print(torch.cuda.device_count())
            cudnn.benchmark = True
            print('Using CUDA..')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)





    def train(self,epoch,trainloader,mixup=False,alpha=0,cutoff=False):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            if mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha, self.use_cuda)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
            outputs = self.net(inputs)
            if mixup:
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                train_loss += loss.data[0]
            else:
                loss =self.criterion(outputs,targets)
                train_loss += loss.data
             
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            if mixup:
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            else:
                correct += predicted.eq(targets.data).cpu().sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if mixup:
                progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
            else:
                progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))

        return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


    def test(self,epoch,testloader):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            self.outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
        acc = 100.*correct/total
        if epoch == start_epoch + self.args.epoch - 1 or acc > best_acc:
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc
        return (test_loss/batch_idx, 100.*correct/total)


    def checkpoint(self,acc, epoch):
    # Save checkpoint.
        print('Saving..')
        state = {
            'net': self.net,
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7' + self.args.name + '_'
               + str(self.args.seed))


    def adjust_learning_rate(self,optimizer, epoch):
        lr = self.args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    

   
