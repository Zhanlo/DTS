import os

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='0'
import time


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import datetime
import json
import collections
import pathlib
import copy
import logging
from matplotlib import pyplot as plt
from tqdm import tqdm

from load_dataset import *
from wideresnet import WideResNet, dual_t_s_WideResNet, dual_t_s_WideResNet_K_plus_one


parser = argparse.ArgumentParser(description='model')
parser.add_argument('--lr', default=0.128, type=float, help='learning rate')
parser.add_argument('--epochs', default=400, type=int, help='Total number of epochs')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset CIFAR10')
parser.add_argument('--batch_size', default=256, type=int, help='Training batch size.')
parser.add_argument('--ts_iteration', default=3, type=int, help='number of student to teacher switch iterations')
parser.add_argument('--n_labels', type=int, default=2400)
parser.add_argument('--n_unlabels', type=int, default=20000)
parser.add_argument('--n_valid', type=int, default=5000)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--ratio', type=float, default=0.6)
parser.add_argument('--file_name', type=str)
parser.add_argument('--lm', type=float, default=0.5)
parser.add_argument('--ls_eps', type=float, default=0.15, help='label smooth')
parser.add_argument('--threshold', type=float, default=0.85)

parser.add_argument('--Ctk_weight', type=float, default=0.25)
parser.add_argument('--Ctu_weight', type=float, default=0.1)

parser.add_argument('--socr_s2_weight', type=float, default=0.5)
parser.add_argument('--uratio', type=int, default=7,
                    help='the ratio of unlabeled data to labeld data in each mini-batch')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def create_model(model_name):
    logging.info('==> Building model..')

    if model_name == 'WideResnet':
        model = WideResNet(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
    elif model_name == 'dual_t_s_WideResNet':
        model = dual_t_s_WideResNet(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
    elif model_name == 'dual_t_s_WideResNet_K_plus_one':
        model = dual_t_s_WideResNet_K_plus_one(widen_factor=2, n_classes=args.n_class, transform_fn=None).to(device)
    return model










def kl_div(p_output, get_softmax=False):

    if get_softmax:
        p_output = F.softmax(p_output)
    uniform_output = p_output*0+(1/args.n_class)
    uniform_output = F.softmax(uniform_output)

    return F.kl_div(p_output, uniform_output, reduction='none')



def compute_probabilities_batch6(out_t, out_t2, lm):
    m = torch.nn.Softmax(dim=-1).cuda()
    msp1 = m(out_t)[:, -1]
    msp2, _ = torch.max(m(out_t2), dim=-1)
    msp2 = 1-msp2
    msp = lm * msp1 + (1-lm)*msp2
    return msp

def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):

    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
        instance_normalize = N
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        instance_normalize = torch.sum(instance_level_weight) + epsilon
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(instance_normalize)



# Training
def train_dual_tea_stu(iter_num, epoch, teacher_0,
                       teacher_1, teacher_2, student_1, student_2,
                       labeled_loader, unlabeled_loader, K_num_class=6,
                       lm=0.8, ls_eps=0.1, threshold=0.85):

    teacher_0.eval()
    teacher_1.eval()
    teacher_2.eval()

    alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch+1) / float(iter_num+1)))))) - 1)

    ts_1_train_loss = []
    ts_2_train_loss = []
    correct_1 = 0
    total_1 = 0
    correct_2 = 0
    total_2 = 0


    labeled_loader = tqdm(labeled_loader)
    labeled_loader.set_description('[%s %04d/%04d]' % ('train', epoch, args.epochs))
    iter_u = iter(unlabeled_loader)

    for batch_idx, (inputs, inputs_noaug, target, dataset_index) in enumerate(labeled_loader):
        student_1.train()
        student_2.eval()
        try:
            inputs_u, inputs_noaug_u, target_u, index_u = next(iter_u)

        except StopIteration:
            iter_u = iter(unlabeled_loader)
            inputs_u, inputs_noaug_u, target_u, index_u = next(iter_u)

        inputs_u = inputs_u.to(device)
        inputs_noaug_u = inputs_noaug_u.to(device)


        with torch.no_grad():
            if iter_num == 0:
                u_logit_K, u_logit_K_plus_one = teacher_0(inputs_noaug_u)
                w_unk_posterior = compute_probabilities_batch6(u_logit_K_plus_one, u_logit_K, lm)
                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(device)
                w_unk_posterior = w_unk_posterior.to(device)
            else:
                u_logit_K = teacher_1(inputs_noaug_u)
                u_logit_K_plus_one = teacher_2(inputs_noaug_u)
                w_unk_posterior = compute_probabilities_batch6(u_logit_K_plus_one, u_logit_K, lm)
                # 获取不确定性得分
                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(device)
                w_unk_posterior = w_unk_posterior.to(device)


        inputs, target = inputs.to(device), target.long().to(device)
        optimizer_student_1.zero_grad()

        outputs_s1_u = student_1(inputs_u)
        outputs_s1_u_noaug = student_1(inputs_noaug_u)

        with torch.no_grad():
            outputs_t1_u_noaug = teacher_1(inputs_noaug_u)
            unlabel_prob_K_t1 = torch.softmax(outputs_t1_u_noaug.detach(), dim=-1)

            max_probs_1, pseudo_label = torch.max(unlabel_prob_K_t1, dim=-1)

        targets_u_onehot_s1 = nn.functional.one_hot(pseudo_label, num_classes=K_num_class)


        mask_s1 = max_probs_1.ge(threshold).float()


        mask_s1_ = w_unk_posterior.le(max_probs_1).float()
        mask_s1 = mask_s1*mask_s1_

        loss_student1_u = CrossEntropyLoss(label=targets_u_onehot_s1,
                                            predict_prob=F.softmax(outputs_s1_u, dim=1),
                                            instance_level_weight=mask_s1)


        outputs_s1_u_kl = F.log_softmax(outputs_s1_u, dim=1)
        loss_KL = F.kl_div(outputs_s1_u_kl, unlabel_prob_K_t1, reduction='none')
        loss_KL = torch.mean(torch.sum(loss_KL, dim=1) * mask_s1)



        outputs_s1_l = student_1(inputs)
        label_s1_onehot = nn.functional.one_hot(target, num_classes = K_num_class)

        label_s1_onehot = label_s1_onehot * (1 - ls_eps)
        label_s1_onehot = label_s1_onehot + ls_eps / (K_num_class)

        loss_ce = CrossEntropyLoss(label=label_s1_onehot, predict_prob=F.softmax(outputs_s1_l, dim=1))

        del label_s1_onehot, outputs_s1_u_noaug

        loss_student_1 = loss_ce + (loss_student1_u + loss_KL) * args.Ctk_weight

        loss_student_1.backward()
        optimizer_student_1.step()
        ts_1_train_loss.append(loss_student_1.item())

        _, predicted = outputs_s1_l.max(1)
        total_1 += target.size(0)
        correct_1 += predicted.eq(target).sum().item()
        total_acc_1 = correct_1/total_1



        student_1.eval()
        student_2.train()
        optimizer_student_2.zero_grad()
        u_outputs_s2 = student_2(inputs_u)
        del outputs_s1_u
        u_outputs_s2_noaug = student_2(inputs_noaug_u)

        label_unknown = (K_num_class) * torch.ones(target_u.size()[0], dtype=torch.long).to(device)
        label_unknown = nn.functional.one_hot(label_unknown, num_classes=(K_num_class+1))
        label_unknown = label_unknown * (1 - ls_eps)
        label_unknown = label_unknown + ls_eps / (K_num_class+1)

        loss_unlabel_s2_K_plus_1 = alpha*CrossEntropyLoss(label=label_unknown, predict_prob=F.softmax(u_outputs_s2, dim=1),
                                                    instance_level_weight=w_unk_posterior)
        del label_unknown,  target_u


        with torch.no_grad():
            outputs_t2_u_noaug = teacher_2(inputs_noaug_u)
            unlabel_prob_K_t2 = torch.softmax(outputs_t2_u_noaug.detach(), dim=-1)
            max_probs, pseudo_label_2 = torch.max(unlabel_prob_K_t2, dim=-1)

        targets_u_onehot = nn.functional.one_hot(pseudo_label_2, num_classes=K_num_class+1)
        mask = max_probs.ge(threshold).float()
        mask_ = w_unk_posterior.le(max_probs).float()
        mask = mask*mask_
        del pseudo_label_2,unlabel_prob_K_t2,outputs_t2_u_noaug


        loss_unlabel_s2_K = CrossEntropyLoss(label=targets_u_onehot,
                                                    predict_prob=F.softmax(u_outputs_s2, dim=1),
                                                    instance_level_weight=mask)

        u_outputs_s2_kl = F.log_softmax(u_outputs_s2, dim=1)

        loss_socr_s2 = F.kl_div(u_outputs_s2_kl, torch.softmax(u_outputs_s2_noaug, dim=1), reduction='batchmean')

        loss_unlabel_s2 = loss_unlabel_s2_K_plus_1 * args.Ctu_weight +\
                          loss_unlabel_s2_K * args.Ctk_weight + loss_socr_s2 * args.socr_s2_weight

        outputs_s2 = student_2(inputs)
        labels_s2_onehot = nn.functional.one_hot(target, num_classes=K_num_class+1)
        labels_s2_onehot = labels_s2_onehot * (1 - ls_eps)
        labels_s2_onehot = labels_s2_onehot + ls_eps / (K_num_class+1)
        loss_ce_2 = CrossEntropyLoss(label=labels_s2_onehot, predict_prob=F.softmax(outputs_s2, dim=1))

        loss_student_2 = loss_ce_2 + loss_unlabel_s2

        loss_student_2.backward()
        optimizer_student_2.step()
        ts_2_train_loss.append(loss_student_2.item())
        _, predicted_2 = outputs_s2.max(1)
        total_2 += target.size(0)
        correct_2 += predicted_2.eq(target).sum().item()
        total_acc_2 = correct_2/total_2

        postfix = {}
        loss_1 = sum(ts_1_train_loss) / len(ts_1_train_loss)
        postfix['loss_student_1'] = loss_1
        postfix['acc_1'] = total_acc_1
        loss_2 = sum(ts_2_train_loss) / len(ts_2_train_loss)
        postfix['loss_student_2'] = loss_2
        postfix['acc_2'] = total_acc_2
        labeled_loader.set_postfix(postfix)
        logging.info('[Training] iter-epoch-batch={}-{}-{}, loss1={:.4f}, acc1={:.4f}, loss2={:.4f}, acc2={:.4f}'\
                          .format(i, epoch, batch_idx, loss_1, total_acc_1, loss_2, total_acc_2))

    total_loss_1 = sum(ts_1_train_loss) / len(ts_1_train_loss)
    total_loss_2 = sum(ts_2_train_loss) / len(ts_2_train_loss)
    total_acc_1 = correct_1 / total_1
    total_acc_2 = correct_2 / total_2

    log = collections.OrderedDict({
        'iter_num':iter_num,
        'epoch': epoch,
        'train':
            collections.OrderedDict({
                'loss_1': total_loss_1,
                'accuracy_1': total_acc_1,
                'loss_2': total_loss_2,
                'accuracy_2': total_acc_2,
            }),
    })
    return log


def test_teacher(iter, total_iter, model_teacher, test_loader):
    model_teacher.eval()

    test_loss = []
    correct = 0
    total = 0
    testloader = tqdm(test_loader)
    testloader.set_description('[%s %04d/%04d]' % ('*test', iter, total_iter))


    with torch.no_grad():
        for batch_idx, (inputs, target, dataset_index) in enumerate(test_loader):

            inputs, target = inputs.to(device), target.long().to(device)
            outputs1 = model_teacher(inputs)
            loss1 = criterion(outputs1, target)

            test_loss.append(loss1.item())

            _, predicted = outputs1.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            total_acc = correct / total

            postfix = {}
            postfix['loss'] = sum(test_loss) / len(test_loss)
            postfix['acc'] = total_acc
            testloader.set_postfix(postfix)

    total_loss_Multi_classification = sum(test_loss) / len(test_loss)
    total_acc_Multi_classification = correct / total
    log_Multi_classification = collections.OrderedDict({
        'iter': iter,
        'test_teacher':
            collections.OrderedDict({
                'loss': total_loss_Multi_classification,
                'accuracy': total_acc_Multi_classification,
            }),
    })
    return log_Multi_classification, total_acc_Multi_classification


def test(epoch, model, testloader, total_epoch):
    global best_acc
    model.eval()

    test_loss = []
    correct = 0
    total = 0
    testloader = tqdm(testloader)
    testloader.set_description('[%s %04d/%04d]' % ('*test', epoch, total_epoch))

    with torch.no_grad():
        for batch_idx, (inputs, target, data_index) in enumerate(testloader):
            inputs, target = inputs.to(device), target.long().to(device)
            outputs1 = model(inputs)
            loss1 = criterion(outputs1, target)

            # test_loss += loss1.item()
            test_loss.append(loss1.item())

            _, predicted = outputs1.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            total_acc = correct / total

            postfix = {}
            postfix['loss'] = sum(test_loss) / len(test_loss)
            postfix['acc'] = total_acc
            testloader.set_postfix(postfix)

    total_loss = sum(test_loss) / len(test_loss)
    total_acc = correct / total
    logging.info('[Testing] epoch/total_epoch={}/{}, loss={:.4f}, acc={:.4f}' \
                 .format(epoch, total_epoch, total_loss, total_acc))

    log = collections.OrderedDict({
        'epoch': epoch,
        'test':
            collections.OrderedDict({
                'loss': total_loss,
                'accuracy': total_acc,
            }),
    })
    return log, total_acc

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset.lower()

    start_time = time.time()

    start_time_formatted = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))

    folder_path = os.path.join(f'results/{args.dataset}/', start_time_formatted)

    os.makedirs(folder_path, exist_ok=True)

    file_name = f'{folder_path}/{start_time_formatted}.log'
    args.file_name = file_name


    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=file_name,
                        level=logging.DEBUG)
    print("logging")

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f'start_time: {start_time_formatted}')



    logging.info(args.__dict__)
    print(args.__dict__)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best eval accuracy
    test_acc = 0
    best_test_acc = 0

    l_train_set, u_train_set, validation_set, test_set = get_dataloaders(dataset=args.dataset,
                                                                                  n_labels=args.n_labels,
                                                                                  n_unlabels=args.n_unlabels,
                                                                                  n_valid=args.n_valid,
                                                                                  tot_class=args.n_class,
                                                                                  ratio=args.ratio)

    labeled_loader = torch.utils.data.DataLoader(l_train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = torch.utils.data.DataLoader(u_train_set, batch_size=args.batch_size * args.uratio, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=2, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              drop_last=False)





    teacher_0 = create_model('dual_t_s_WideResNet')#
    teacher_1 = create_model('WideResnet')
    student_1 = create_model('WideResnet')
    teacher_2 = create_model('dual_t_s_WideResNet_K_plus_one')
    student_2 = create_model('dual_t_s_WideResNet_K_plus_one')


    teacher_0 = teacher_0.to(device)
    teacher_1 = teacher_1.to(device)
    student_1 = student_1.to(device)
    teacher_2 = teacher_2.to(device)
    student_2 = student_2.to(device)
    cudnn.benchmark = True


    # optimizer
    optimizer_teacher_0 = optim.SGD(teacher_0.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True,
                                    dampening=0)
    scheduler_teacher_0 = torch.optim.lr_scheduler.StepLR(optimizer_teacher_0, step_size=5, gamma=0.97)

    optimizer_teacher_1 = optim.SGD(teacher_1.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True,
                                    dampening=0)
    scheduler_teacher_1 = torch.optim.lr_scheduler.StepLR(optimizer_teacher_1, step_size=5, gamma=0.97)

    optimizer_teacher_2 = optim.SGD(teacher_2.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True,
                                    dampening=0)
    scheduler_teacher_2 = torch.optim.lr_scheduler.StepLR(optimizer_teacher_2, step_size=5, gamma=0.97)

    optimizer_student_1 = optim.SGD(student_1.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True,
                                    dampening=0)
    scheduler_student_1 = torch.optim.lr_scheduler.StepLR(optimizer_student_1, step_size=5, gamma=0.97)

    optimizer_student_2 = optim.SGD(student_2.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True,
                                    dampening=0)
    scheduler_student_2 = torch.optim.lr_scheduler.StepLR(optimizer_student_2, step_size=5, gamma=0.97)


    criterion = nn.CrossEntropyLoss()






    logging.info('load the pretrain model')
    pretrain_dict = torch.load(os.path.join("./save_model/", f"pretrain_teacher_dual_t_s_WideResnet_{dataset_name}.pth"))
    teacher_0.load_state_dict(pretrain_dict)
    logging.info(f'pretrain model load_path{os.path.join("./save_model/", f"pretrain_teacher_dual_t_s_WideResnet_{dataset_name}.pth")}')

    model_dict = teacher_1.state_dict()

    updated_dict_1 = {k: v for k, v in pretrain_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    model_dict.update(updated_dict_1)
    teacher_1.load_state_dict(model_dict, strict=True)
    student_1.load_state_dict(model_dict, strict=True)


    model_dict_2 = teacher_2.state_dict()
    updated_dict_2 = {k: v for k, v in pretrain_dict.items() if k in model_dict_2 and v.size() == model_dict_2[k].size()}

    model_dict_2.update(updated_dict_2)
    teacher_2.load_state_dict(model_dict_2, strict=True)
    student_2.load_state_dict(model_dict_2, strict=True)
    logging.info('load the pretrain model --done')

    train_loss_1_list = []
    test_loss_1_list = []
    valid_loss_1_list = []
    train_loss_2_list = []
    test_loss_2_list = []
    valid_loss_2_list = []


    train_acc_1_list = []
    test_acc_1_list = []
    valid_acc_1_list = []
    train_acc_2_list = []
    test_acc_2_list = []
    valid_acc_2_list = []


    test_teacher_acc = []
    test_acc_1_epoch = []
    test_teacher_acc_1 = []
    test_teacher_acc_2 = []



    K_num_class = args.__dict__['n_class']
    lm = args.__dict__['lm']
    ls_eps = args.__dict__['ls_eps']
    threshold = args.__dict__['threshold']


    for i in range(args.ts_iteration):

        logging.info('\n[{}/{}] iterative training on student'.format(i + 1, args.ts_iteration))


        for epoch in range(args.epochs):
            # logging.info('start train------')
            # args.epochs=400
            train_dual_tea_stu_log = train_dual_tea_stu(i, epoch, teacher_0, teacher_1, teacher_2, student_1, student_2,
                                                        labeled_loader, unlabeled_loader, K_num_class, lm, ls_eps, threshold)


            train_loss_1_list.append(train_dual_tea_stu_log['train']['loss_1'])
            train_loss_2_list.append(train_dual_tea_stu_log['train']['loss_2'])
            train_acc_1_list.append(train_dual_tea_stu_log['train']['accuracy_1'])
            train_acc_2_list.append(train_dual_tea_stu_log['train']['accuracy_2'])


            if epoch % 10 == 0 and epoch != 0:

                valid_log, acc = test(epoch, student_1, validation_loader, args.epochs)
                valid_loss_1_list.append(valid_log['test']['loss'])
                valid_acc_1_list.append(valid_log['test']['accuracy'])

                if (acc > best_acc):
                    best_acc = acc
                    test_log, test_acc = test(epoch, student_1, test_loader, args.epochs)
                    test_loss_1_list.append(test_log['test']['loss'])
                    test_acc_1_list.append(test_log['test']['accuracy'])
                    test_acc_1_epoch.append(epoch)
                    if test_log['test']['accuracy']>best_test_acc:
                        best_test_acc=test_log['test']['accuracy']

                    torch.save(student_1.state_dict(), os.path.join("./save_model/", f'{dataset_name}_student_1.pth'))

            scheduler_student_1.step()
            scheduler_student_2.step()

        if i!=args.ts_iteration:
            log, acc = test_teacher(i + 1, args.ts_iteration + 1, teacher_1, test_loader)
            test_teacher_acc_1.append(acc)


        if i != args.ts_iteration - 1:

            teacher_1 = create_model('WideResnet')
            cudnn.benchmark = True
            save_path = f'{dataset_name}_student_1.pth'

            teacher_1.load_state_dict(
                torch.load(os.path.join("./save_model/", save_path)))
            teacher_1 = teacher_1.to(device)

            optimizer_student_1 = optim.SGD(student_1.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                            nesterov=True,
                                            dampening=0)
            scheduler_student_1 = torch.optim.lr_scheduler.StepLR(optimizer_student_1, step_size=5, gamma=0.97)


            torch.save(student_2.state_dict(), os.path.join("./save_model/", f'{dataset_name}_student_2.pth'))
            teacher_2 = create_model('dual_t_s_WideResNet_K_plus_one')
            save_path = f'{dataset_name}_student_2.pth'
            teacher_2.load_state_dict(
                torch.load(os.path.join("./save_model/", save_path)))
            teacher_2 = teacher_2.to(device)

            optimizer_student_2 = optim.SGD(student_2.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                            nesterov=True,
                                            dampening=0)
            scheduler_student_2 = torch.optim.lr_scheduler.StepLR(optimizer_student_2, step_size=5, gamma=0.97)



    lists = [train_loss_1_list, train_loss_2_list, test_loss_1_list, valid_loss_1_list,
             train_acc_1_list, valid_acc_1_list]
    labels = ['train_loss_student-1', 'train_loss_student-2', 'test_loss_student-1', 'valid_loss_student-1',
              'train_acc_student-1', 'valid_acc_student-1']


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
        plt.savefig(f'{folder_path}/{start_time_formatted}_{dataset_name}_{label}.png')
        plt.close()

    lists = [test_teacher_acc_1, test_acc_1_list, test_acc_1_epoch]
    labels = ['test_teacher_acc_1', 'test_acc_student-1', 'test_acc_student_1_epoch']

    for lst, label in zip(lists, labels):
        plt.figure()
        plt.plot(lst, label=label, marker='o')
        for i, value in enumerate(lst):
            plt.annotate(round(value, 4), (i, value))

        plt.ylabel(label)
        plt.xlabel('iter')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/{start_time_formatted}_{dataset_name}_{label}.png')
        plt.close()
        logging.info(label)
        logging.info(lst)




    name = [str(start_time_formatted)]

    setting = [dataset_name, f'ratio: {args.ratio}',
               f'socr_weight: {args.socr_s2_weight}',
               f'iteration: {args.ts_iteration}',
               f'ctk: {args.Ctk_weight}',
               f'ctu: {args.Ctu_weight}',
               f"lm: {args.lm}",
               f"threshold: {args.threshold}"]
    lists = [test_teacher_acc_1, test_acc_1_list[-4:], best_test_acc, setting, name]
    labels = ['test_teacher_acc_1', 'test_acc_student-1', 'best_acc',
               'setting', 'start_time_formatted']

    data_dict = dict(zip(labels, lists))
    for key in data_dict:
        data_dict[key] = [str(data_dict[key])]


    df_single_element = pd.DataFrame(data_dict)
    csv_path = f"results/csv/cifar100.csv"

    directory = os.path.dirname(csv_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


    if os.path.exists(csv_path):
        df_single_element.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_single_element.to_csv(csv_path, index=False)

    end_time = time.time()
    logging.info("Total time: %f" % (end_time - start_time))

