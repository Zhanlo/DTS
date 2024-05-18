import os
import time

from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
import datetime
import json
import collections
import pathlib
import copy
from tqdm import tqdm
# from utils import *
from load_dataset import *
import os
from wideresnet import WideResNet, dual_t_s_WideResNet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description='Model')
parser.add_argument('--lr', default=0.128, type=float, help='learning rate')
parser.add_argument('--warm_up', default=1000, type=int, help='number of epochs before main training starts')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset CIFAR10')
parser.add_argument('--model', default='WideResnet', type=str, help='WideResnet')
parser.add_argument('--batch_size', default=256, type=int, help='Training batch size.')
parser.add_argument('--ts_iteration', default=3, type=int, help='number of student to teacher switch iterations')
parser.add_argument('--n_labels', type=int, default=2400)
parser.add_argument('--n_unlabels', type=int, default=20000)
parser.add_argument('--n_valid', type=int, default=5000)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--ratio', type=float, default=0.6)
parser.add_argument('--name', default='model', type=str, help='Name of the experiment')

def create_model(model_name):
    logging.info('==> Building model..')
    if model_name == 'WideResnet':
        model = WideResNet(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
        print("create WideResnet---")
    elif model_name == 'dual_t_s_WideResnet':

        model = dual_t_s_WideResNet(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
        print('create dual_t_s_WideResnet---')
    return model

def warmup(epoch, model, trainloader):
    model.train()

    _train_loss= []
    correct_1 = 0
    total_1 = 0

    correct_2 = 0
    total_2 = 0

    trainloader = tqdm(trainloader)

    trainloader.set_description('[%s %04d/%04d]' % ('warmup', epoch, args.warm_up))

    for batch_idx, (inputs, inputs_noaug, target, dataset_index) in enumerate(trainloader):

        inputs, target = inputs.to(device), target.long().to(device)
        optimizer_teacher.zero_grad()
        outputs1, outputs2 = model(inputs)
        loss_1 = criterion(outputs1, target)
        loss_2 = criterion(outputs2, target)
        loss = loss_1 + loss_2

        loss.backward()
        optimizer_teacher.step()


        _train_loss.append(loss.item())
        _, predicted = outputs1.max(1)

        total_1 += target.size(0)
        correct_1 += predicted.eq(target).sum().item()

        total_acc_1 = correct_1 / total_1

        total_2 += target.size(0)
        _, predicted_2 = outputs2.max(1)
        correct_2 += predicted_2.eq(target).sum().item()
        total_acc_2 = correct_2 / total_2

        postfix = {}
        loss = sum(_train_loss)/len(_train_loss)
        postfix['loss'] = loss
        postfix['acc_K'] = total_acc_1
        postfix['acc_K_plus_1'] = total_acc_2

        postfix['lr'] = optimizer_teacher.param_groups[0]['lr']
        trainloader.set_postfix(postfix)
        logging.info('[Warmup] epoch/total_epochs={}/{}, loss={:.4f}, acc_K={:.4f}, acc_K+1={:.4f}' \
                     .format(epoch, args.warm_up, loss, total_acc_1, total_acc_2))

    total_loss = sum(_train_loss)/len(_train_loss)
    total_acc_1 = correct_1 / total_1
    total_acc_2 = correct_2 / total_2

    log = collections.OrderedDict({
        'epoch': epoch,
        'train':
            collections.OrderedDict({
                'loss': total_loss,
                'acc_K': total_acc_1,
                'acc_K+1': total_acc_2
            }),
    })
    return log

def test(epoch, model, testloader, total_epoch):
    global best_acc
    model.eval()

    _test_loss = []
    correct_1 = 0
    total_1 = 0
    correct_2 = 0
    total_2 = 0
    testloader = tqdm(testloader)
    testloader.set_description('[%s %04d/%04d]' % ('*test', epoch, total_epoch))

    with torch.no_grad():
        for batch_idx, (inputs, target, data_index) in enumerate(testloader):
            inputs, target = inputs.to(device), target.long().to(device)
            outputs1, outputs2 = model(inputs)
            loss1 = criterion(outputs1, target)
            loss2 = criterion(outputs2, target)
            loss = loss1 + loss2
            _test_loss.append(loss.item())

            _, predicted = outputs1.max(1)
            total_1 += target.size(0)
            correct_1 += predicted.eq(target).sum().item()
            total_acc_1 = correct_1 / total_1

            _, predicted_2 = outputs2.max(1)
            total_2 += target.size(0)
            correct_2 += predicted_2.eq(target).sum().item()
            total_acc_2 = correct_2 / total_2

            postfix = {}
            postfix['loss'] = sum(_test_loss) / len(_test_loss)
            postfix['acc_K'] = total_acc_1
            postfix['acc_K+1'] = total_acc_2
            testloader.set_postfix(postfix)

    total_loss = sum(_test_loss) / len(_test_loss)
    total_acc_1 = correct_1 / total_1
    total_acc_2 = correct_2 / total_2
    logging.info('[Warmup] epoch/total_epochs={}/{}, loss={:.4f}, acc_K={:.4f}, acc_K+1={:.4f}' \
                 .format(epoch, args.warm_up, total_loss, total_acc_1, total_acc_2))

    log = collections.OrderedDict({
        'epoch': epoch,
        'test':
            collections.OrderedDict({
                'loss': total_loss,
                'acc_K': total_acc_1,
                'acc_K+1': total_acc_2
            }),
    })
    return log, total_acc_1, total_acc_2




if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset.lower()

    start_time = time.time()

    start_time_formatted = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    folder_path = os.path.join(f'results/{args.dataset}/', start_time_formatted)

    os.makedirs(folder_path, exist_ok=True)

    file_name = f'{folder_path}/pretrain_{start_time_formatted}.log'
    args.file_name = file_name

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=file_name,
                        level=logging.DEBUG)
    logging.info('logging')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f'start_time: {start_time_formatted}')

    logging.info(args.__dict__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy


    l_train_set, u_train_set, validation_set, test_set = get_dataloaders(dataset=args.dataset,
                                              n_labels=args.n_labels,
                                              n_unlabels=args.n_unlabels,
                                              n_valid=args.n_valid,
                                              tot_class=args.n_class,
                                              ratio=args.ratio)


    labeled_loader = torch.utils.data.DataLoader(l_train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    unlabeled_loader = torch.utils.data.DataLoader(u_train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=2, drop_last=False)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              drop_last=False)



    model_student = create_model('dual_t_s_WideResnet')
    model_teacher = create_model('dual_t_s_WideResnet')


    model_student = model_student.to(device)
    print("==================================")
    model_teacher = model_teacher.to(device)


    cudnn.benchmark = True
    optimizer_teacher = optim.SGD(model_teacher.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                  nesterov=True,
                                  dampening=0)
    scheduler_teacher = torch.optim.lr_scheduler.StepLR(optimizer_teacher, step_size=5, gamma=0.97)

    optimizer_student = optim.SGD(model_student.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                  nesterov=True,
                                  dampening=0)
    scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=5, gamma=0.97)

    criterion = nn.CrossEntropyLoss()



    acc_1_list = []
    acc_2_list = []
    train_loss_list = []
    train_acc_1_list = []
    train_acc_2_list = []
    test_loss_list = []

    for epoch in range(args.warm_up):
        train_log = warmup(epoch, model_teacher, labeled_loader)
        exp_log = train_log.copy()
        train_loss_list.append(train_log['train']['loss'])
        train_acc_1_list.append(train_log['train']['acc_K'])
        train_acc_2_list.append(train_log['train']['acc_K+1'])
        if epoch % 10 == 0 and epoch != 0:
            test_log, acc_1, acc_2 = test(epoch, model_teacher, validation_loader, args.warm_up)
            acc_1_list.append(acc_1)
            acc_2_list.append(acc_2)
            test_loss_list.append(test_log['test']['loss'])
            exp_log.update(test_log)
            if (acc_1 > best_acc):
                best_acc = acc_1
                logging.info(f'best_acc:{best_acc}')
                torch.save(model_teacher.state_dict(),
                           os.path.join("./save_model/",
                                        f"pretrain_teacher_dual_t_s_WideResnet_{dataset_name}.pth"))
                logging.info(f'pretrain model save_path{os.path.join("./save_model/", f"pretrain_teacher_dual_t_s_WideResnet_{dataset_name}.pth")}')
        scheduler_teacher.step()


    lists = [acc_1_list, acc_2_list, test_loss_list, train_loss_list, train_acc_1_list, train_acc_2_list]
    labels = ['test_acc_K', 'test_acc_K+1', 'test_loss', 'train_loss', 'train_acc_K', 'train_acc_K+1']
    for lst, label in zip(lists, labels):
        plt.figure()
        plt.plot(lst, label=label, marker='o')

        max_value = max(lst)
        max_index = lst.index(max_value)
        min_value = min(lst)
        min_index = lst.index(min_value)
        plt.annotate(round(max_value,4),(max_index,max_value))
        plt.annotate(round(min_value, 4), (min_index, min_value))

        plt.ylabel(label)
        if label == 'test_teacher-1_acc':
            plt.xlabel('iter_test_teacher-1_acc')
        else:
            plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/pretrain_{dataset_name}_{int(start_time)}_{label}.png')
        plt.close()
        if 'test' in label:
            logging.info(label)
            logging.info(lst)
            print(label)
            print(lst)