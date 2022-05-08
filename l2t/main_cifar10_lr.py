import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


from network.teacher_model.teacher_mlp import teacher_mlp_lr
from network.student_model.resnet import resnet32, _weights_init
from dataset.data_loader import data_loader_func
from utils.setseed import setup_seed
from scipy.stats import rankdata
import math
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import time
import numpy as np
from utils.reset_lr import group_wise_lr, create_configs_predefined

from collections import deque
from pathlib import Path


np.set_printoptions(threshold=np.inf)


def state_func(state_config, total_layers=4):
    assert total_layers > 0 and type(total_layers) is int
    max_loss = state_config['max_loss']
    avg_val_acc = state_config['avg_val_acc']
    avg_train_loss = state_config['avg_train_loss']

    state_feactues = torch.zeros((total_layers, 3)).to(device)
    # print(state_feactues.dtype)
    # print(avg_val_acc)
    # print(average_train_loss)
    state_feactues[range(total_layers), 0] = avg_train_loss / max_loss
    state_feactues[range(total_layers), 1] = avg_val_acc
    state_feactues[range(total_layers), 2] = torch.from_numpy(
        rankdata(np.arange(total_layers)) / total_layers).to(state_config['device'], dtype=torch.float)

    return state_feactues


def train_student(config):
    model = config['model']
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    inputs = config['inputs'].to(config['device'])
    labels = config['labels'].to(config['device'])

    # zero the parameter gradients
    optimizer = config['optimizer']
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    total += inputs.size(0)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(predicted == labels.data)

    train_loss = running_loss / total
    train_acc = running_corrects.double() / total

    return train_loss, train_acc


def val_student(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            # statistics
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

        val_loss = running_loss / total
        val_acc = running_corrects.double() / total

    return val_loss, val_acc.item()


def student_lr_scheduler_cifar10(optimizer, iterations):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations == 32000:
        lr = lr * 0.1
        print('Adjust learning rate to : ', lr)
    elif iterations == 48000:
        lr = lr * 0.1
        print('Adjust learning rate to : ', lr)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def teacher_lr_scheduler(optimizer, iterations):
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations % 50 == 0:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_teacher_by_optimizer(optimizer, device):
    tensor_reward = []
    loss_values = []
    rewards = teacher_model.rewards
    normalized_reward = []

    mean = torch.mean(torch.tensor(rewards))
    for i in range(len(rewards)):
        normalized_reward.append(rewards[i] - mean)

    for r in normalized_reward:
        tensor_reward.insert(int(r.item()), normalized_reward[int(r.item())])
    tensor_reward = torch.tensor(tensor_reward).to(device)
    for log_prob, reward in zip(teacher_model.saved_log_probs, tensor_reward):
        r = -log_prob * reward

        loss_values.append(r)
    optimizer.zero_grad()
    loss_values_f = torch.stack(loss_values)
    loss = loss_values_f.mean()
    print(f'Outer loss is {loss} ')
    loss.backward()
    optimizer.step()
    teacher_model.saved_log_probs = []
    teacher_model.rewards = []


def select_action(state):
    action_prb = teacher_model(state.detach())
    m = Categorical(action_prb)
    action = m.sample()
    teacher_model.saved_log_probs.append(m.log_prob(action))
    return action


def check_if_decreasing(acc_list):
    out = np.array(acc_list) - acc_list[-1]
    if (out[:-1] < 0).sum() == 0:
        return True
    return False


def train_l2t(teacher_model, student_model):
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)
    best_avg_val_acc = 0.
    best_student_model = copy.deepcopy(student_model)

    best_avg_training_loss = config['max_loss']

    # lr_current = torch.tensor([0.001] * len(layer_list), dtype=torch.float32)
    lr_last = torch.ones(len(layer_list)) * 0.01
    lr_best = lr_last

    best_avg_val_acc_for_episode_queue = deque(maxlen=5)

    for i_episode in trange(config['train_episode']):

        # compute the state feature
        state_config = {
            'avg_train_loss': best_avg_training_loss,
            'avg_val_acc': best_avg_val_acc,
            'max_loss': config['max_loss'],
            'max_iter': config['max_iter'],
            'device': device
        }

        state = state_func(state_config, total_layers=len(layer_list))
        initialized_student_model_this_traj = copy.deepcopy(best_student_model)

        for i, lr in enumerate(lr_last):
            writer.add_scalar(f'train_teacher/lr{i}', lr.item(), i_episode)

        for i_traj in trange(config['traj_num_per_episode']):
            # for each trajectory, reinitialize student model with the previous best student model
            student_model = copy.deepcopy(initialized_student_model_this_traj)

            # the teacher select action according to the state feature
            action = select_action(state)

            # 0 means decrease learning rate, 1 means increase learning rate
            lr_current = lr_last + ((2 * action - 1) * config['lr_increment']).cpu()
            print('Episode: {}, Trajectory: {}, the lr before clamp is {}'.format(
                    i_episode, i_traj, lr_current), file=text_log_file, flush=True)
            lr_current.clamp_(min=0.00001, max=0.1)
            print('Episode: {}, Trajectory: {}, the lr after clamp is {}'.format(
                    i_episode, i_traj, lr_current), file=text_log_file, flush=True)

            lr_configs = create_configs_predefined(layer_list, lr_list=lr_current.numpy().tolist())
            confs, names = group_wise_lr(student_model, lr_configs)

            optimizer = optim.SGD(confs, lr=0.001, momentum=0.9, weight_decay=1e-4)

            # initialize states statistics
            training_loss_history = deque(maxlen=100)
            val_acc_history = []
            student_updates = 0

            for i_epoch in trange(config['epoch_num_per_traj']):
                for idx, (inputs, labels) in enumerate(dataloader['teacher_train_loader']):
                    train_student_config = {
                        'inputs': inputs,
                        'labels': labels,
                        'model': student_model,
                        'device': device,
                        'optimizer': optimizer
                    }

                    # train the student
                    train_loss, train_acc = train_student(train_student_config)

                    training_loss_history.append(train_loss)
                    student_updates += 1

                    print(
                        'Episode: {}, Trajectory: {}, Epoch: {}, Student_iter: {}, Train loss: {:.4f}, Train acc: {:.4f}'.format(
                            i_episode, i_traj, i_epoch, student_updates, train_loss, train_acc))

                # val the student on validation set
                val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
                val_acc_history.append(val_acc)

            # at the end of training each trajectory, collect training supervision for the teacher
            avg_val_acc = sum(val_acc_history) / len(val_acc_history)
            teacher_model.rewards.append(avg_val_acc)

            print('Episode: {}, Trajectory: {}, Avg Val acc: {:.4f}'.format(i_episode, i_traj, avg_val_acc),
                  file=text_log_file, flush=True)

            # update best states
            if avg_val_acc > best_avg_val_acc:
                best_student_model = copy.deepcopy(student_model)
                best_avg_val_acc = avg_val_acc
                best_avg_training_loss = sum(training_loss_history) / len(training_loss_history)
                lr_best = lr_current

        best_avg_val_acc_for_episode_queue.append(best_avg_val_acc)

        writer.add_scalar(f'train_teacher/best_avg_val_acc', best_avg_val_acc, i_episode)

        # after training multiple trajectories, update teacher and update lr
        update_teacher_by_optimizer(teacher_optimizer, device)
        lr_last = lr_best

        torch.save(student_model.state_dict(), os.path.join(config['student_save_dir'], 'student_best.pth'))
        torch.save(teacher_model.state_dict(), os.path.join(config['teacher_save_dir'], 'teacher.pth'))

        print('After episode: {}, the avg training loss is {}, the avg val accuracy is {}, the best lr is {}'.format(
            i_episode, best_avg_training_loss, best_avg_val_acc, lr_best), file=text_log_file, flush=True)

        if len(best_avg_val_acc_for_episode_queue) >= config['max_non_increasing_steps']:
            if check_if_decreasing(best_avg_val_acc_for_episode_queue):
                break

    print("End of training teacher!", file=text_log_file, flush=True)


def test_l2t(teacher_model, student_model):
    teacher_model.eval()
    # init the student
    student_model.apply(_weights_init)

    best_avg_val_acc = 0.
    best_student_model = copy.deepcopy(student_model)
    best_avg_training_loss = config['max_loss']

    # lr_current = torch.tensor([0.001] * len(layer_list), dtype=torch.float32)
    lr_last = torch.ones(len(layer_list)) * 0.01
    lr_best = lr_last

    non_increasing_step = 0

    for i_episode in trange(config['test_episode']):
        # compute the state feature
        state_config = {
            'avg_train_loss': best_avg_training_loss,
            'avg_val_acc': best_avg_val_acc,
            'max_loss': config['max_loss'],
            'max_iter': config['max_iter'],
            'device': device
        }

        state = state_func(state_config, total_layers=len(layer_list))
        # the teacher select action according to the state feature
        action = select_action(state)

        # 0 means decrease learning rate, 1 means increase learning rate
        lr_current = lr_last + ((2 * action - 1) * config['lr_increment']).cpu()
        print('Episode: {}, the lr before clamp is {}'.format(
            i_episode, lr_current), file=text_log_file, flush=True)
        lr_current.clamp_(min=0.00001, max=0.1)
        print('Episode: {}, the lr after clamp is {}'.format(
            i_episode, lr_current), file=text_log_file, flush=True)

        for i, lr in enumerate(lr_current):
            writer.add_scalar(f'train_student/lr{i}', lr.item(), i_episode)

        lr_configs = create_configs_predefined(layer_list, lr_list=lr_current.numpy().tolist())
        confs, names = group_wise_lr(student_model, lr_configs)

        optimizer = optim.SGD(confs, lr=0.001, momentum=0.9, weight_decay=1e-4)

        # initialize states statistics
        training_loss_history = deque(maxlen=100)
        val_acc_history = []
        student_updates = 0

        for i_epoch in trange(config['epoch_num_per_traj']):
            for idx, (inputs, labels) in enumerate(dataloader['student_train_loader']):
                train_student_config = {
                    'inputs': inputs,
                    'labels': labels,
                    'model': student_model,
                    'device': device,
                    'optimizer': optimizer
                }

                # train the student
                train_loss, train_acc = train_student(train_student_config)

                training_loss_history.append(train_loss)
                student_updates += 1

                print(
                    'Episode: {}, Epoch: {}, Student_iter: {}, Train loss: {:.4f}, Train acc: {:.4f}'.format(
                        i_episode, i_epoch, student_updates, train_loss, train_acc))

                # val the student on validation set
            val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
            val_acc_history.append(val_acc)

        # at the end of training each trajectory, collect training supervision for the teacher
        avg_val_acc = sum(val_acc_history) / len(val_acc_history)

        print('Episode: {}, Avg Val acc: {:.4f}'.format(i_episode, avg_val_acc), file=text_log_file, flush=True)

        # update best states
        if avg_val_acc > best_avg_val_acc:
            best_student_model = copy.deepcopy(student_model)
            best_avg_val_acc = avg_val_acc
            best_avg_training_loss = sum(training_loss_history) / len(training_loss_history)
            lr_best = lr_current
            non_increasing_step = 0
        else:
            non_increasing_step += 1

        writer.add_scalar(f'train_student/best_avg_val_acc', best_avg_val_acc, i_episode)

        torch.save(best_student_model.state_dict(), os.path.join(config['student_save_dir'], 'student_best_test.pth'))

        print('After episode: {}, the avg training loss is {}, the avg val accuracy is {}, the best lr is {}'.format(
            i_episode, best_avg_training_loss, best_avg_val_acc, lr_best), file=text_log_file, flush=True)

        if non_increasing_step > 5:
            print(f"Non decreasing for 5 episodes! Stop training now after episode {i_episode}", file=text_log_file,
                  flush=True)
            break

    print("End of training student!", file=text_log_file, flush=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to teach')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_episode', type=int, default=300)
    parser.add_argument('--traj_num_per_episode', type=int, default=8)
    parser.add_argument('--epoch_num_per_traj', type=int, default=20)
    parser.add_argument('--lr_increment', type=float, default=0.0002)
    parser.add_argument('--test_epoch', type=int, default=300)
    parser.add_argument('--exp_name', type=str, default='cifar10_adjust_lr')
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()

    # set seed
    setup_seed(args.seed)

    # create folders
    exp_dir = Path(args.save_dir, args.exp_name)
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)
    teacher_model_save_dir = exp_dir.joinpath('teacher')
    if not teacher_model_save_dir.exists():
        teacher_model_save_dir.mkdir(exist_ok=True)

    student_model_save_dir = exp_dir.joinpath('student')
    if not student_model_save_dir.exists():
        student_model_save_dir.mkdir(exist_ok=True)

    log_dir = Path(args.log_dir, args.exp_name)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'lr_increment': args.lr_increment,
        'traj_num_per_episode': args.traj_num_per_episode,
        'epoch_num_per_traj': args.epoch_num_per_traj,
        'max_iter': 10000,
        'batch_size': args.batch_size,
        'max_non_increasing_steps': 5,
        'num_classes': 10,
        'max_loss': 15,
        'train_episode': args.train_episode,
        'test_episode': args.test_epoch,
        'path_to_dataset': './data',
        'tensorboard_save_path': log_dir,
        'teacher_save_dir': teacher_model_save_dir,
        'teacher_save_model': 'teacher_step1_cifar10.pth',
        'student_save_dir': student_model_save_dir,
        'student_save_model': 'student_step2_cifar10.pth',
        'log_dirs': log_dir,
    }

    text_log_file = open(exp_dir.joinpath('log.txt'), 'a')
    writer = SummaryWriter(config['tensorboard_save_path'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_mlp_lr().to(device)
    student_model = resnet32().to(device)

    for name, param in student_model.named_parameters():
        print(name)

    layer_list = ['layer1', 'layer2', 'layer3']  # TODO: add in contents

    lr_config_list = create_configs_predefined(layer_list, lr_list=[0.001] * len(layer_list))

    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset=config['path_to_dataset'])

    print('Training the teacher starts....................')
    start = time.time()
    train_l2t(teacher_model, student_model)
    time_train = time.time() - start

    torch.save(teacher_model.state_dict(), os.path.join(teacher_model_save_dir, config['teacher_save_model']))

    print('Done.\nTesting the teacher.....................')
    print('Loading the teacher model')
    teacher_model.load_state_dict(torch.load(os.path.join(teacher_model_save_dir, config['teacher_save_model'])))
    start = time.time()
    test_l2t(teacher_model, student_model)
    time_test = time.time() - start

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_train//3600, time_train // 60, time_train % 60))
    print('Testing complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_test // 3600, time_test // 60, time_test % 60))

    print('Saving the student mdoel.......................')
    torch.save(student_model.state_dict(), os.path.join(student_model_save_dir, config['student_save_model']))
