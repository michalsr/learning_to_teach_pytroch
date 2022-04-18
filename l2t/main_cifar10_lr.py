import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


from network.teacher_model.teacher_mlp import teacher_mlp
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


np.set_printoptions(threshold=np.inf)


def state_func(state_config, total_layers=4):
    assert total_layers > 0 and type(total_layers) is int
    max_loss = state_config['max_loss']
    avg_val_acc = state_config['avg_val_acc']
    avg_train_loss = state_config['avg_train_loss']

    state_feactues = torch.zeros((total_layers, 3)).to(device)
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

    return val_loss, val_acc


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


def update_teacher_by_optimizer(optimizer):
    tensor_reward = []
    loss_values = []
    rewards = teacher_model.rewards
    normalized_reward = []

    mean = np.mean(rewards)
    for i in range(len(rewards)):
        normalized_reward.append(rewards[i] - mean)

    for r in normalized_reward:
        tensor_reward.insert(int(r.item()), normalized_reward[int(r.item())])
    tensor_reward = torch.tensor(tensor_reward)
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


def train_l2t():
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)
    best_avg_val_acc = 0.
    best_student_model = copy.deepcopy(student_model)

    best_avg_training_loss = config['max_loss']

    lr_current = torch.tensor([0.001] * len(layer_list), dtype=torch.float32)
    lr_last = lr_current

    for i_episode in trange(config['train_episode']):

        # compute the state feature
        state_config = {
            'avg_train_loss': best_avg_training_loss,
            'avg_val_acc': best_avg_val_acc,
            'max_loss': config['max_loss'],
            'max_iter': config['max_iter'],
            'device': device
        }

        state = state_func(state_config)

        for i_traj in trange(config['traj_num_per_episode']):
            # for each trajectory, reinitialize student model with the previous best student model
            student_model = copy.deepcopy(best_student_model)

            # the teacher select action according to the state feature
            action = select_action(state)

            # 0 means decrease learning rate, 1 means increase learning rate
            lr_current = lr_last + ((2 * action - 1) * config['lr_increment']).cpu()

            lr_configs = create_configs_predefined(layer_list, lr_list=lr_current.numpy().tolist())
            confs, names = group_wise_lr(student_model, lr_configs)

            optimizer = optim.SGD(confs, lr=0.1, momentum=0.9, weight_decay=1e-4)

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
            teacher_model.reward_T_histtory.append(avg_val_acc)

            print('Episode: {}, Trajectory: {}, Avg Val acc: {:.4f}'.format(i_episode, i_traj, avg_val_acc))

            # update best states
            if avg_val_acc > best_avg_val_acc:
                best_student_model = copy.deepcopy(student_model)
                best_avg_val_acc = avg_val_acc
                best_avg_training_loss = sum(training_loss_history) / len(training_loss_history)

        # after training multiple trajectories, update teacher and update lr
        update_teacher_by_optimizer(teacher_optimizer)
        lr_last = lr_current


def test_l2t():
    training_loss_history = []
    best_loss_on_dev = config['max_loss']
    student_updates = 0
    i_iter = 0
    input_pool = []
    label_pool = []
    done = False
    count_sampled = 0
    num_effective_data = 0
    test_acc_list = []
    # init the student
    student_model.apply(_weights_init)

    for epoch in trange(config['test_episode']):
        for idx, (inputs, labels) in enumerate(dataloader['student_train_loader']):

            # compute the state feature
            state_config = {
                'inputs': inputs.to(device),
                'labels': labels.to(device),
                'num_class': config['num_classes'],
                'student_iter': student_updates,
                'training_loss_history': training_loss_history,
                'best_loss_on_dev': best_loss_on_dev,
                'model': student_model,
                'max_loss': config['max_loss'],
                'max_iter': config['max_iter'],
                'device': device
            }
            state = state_func(state_config)

            # the teacher select action according to the state feature
            action = select_action(state)

            # finish one step
            i_iter += 1

            # collect data to train student
            indices = torch.nonzero(action)
            if len(indices) == 0:
                continue

            count_sampled += len(indices)
            selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
            selected_labels = labels[indices.squeeze()].view(-1, 1)
            input_pool.append(selected_inputs)
            label_pool.append(selected_labels)

            if count_sampled >= config['batch_size']:
                inputs = torch.cat(input_pool, 0)[:config['batch_size']]
                labels = torch.cat(label_pool, 0)[:config['batch_size']].squeeze()

                input_pool = []
                label_pool = []
                count_sampled = 0

                train_student_config = {
                    'inputs': inputs,
                    'labels': labels,
                    'model': student_model,
                    'device': device,
                    'optimizer': optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9,
                                           weight_decay=0.0001)
                }

                # train the student
                train_loss, train_acc = train_student(train_student_config)
                training_loss_history.append(train_loss)
                student_updates += 1
                num_effective_data += inputs.size(0)
                student_lr_scheduler_cifar10(train_student_config['optimizer'], student_updates)

                # val the student on validation set
                val_loss, val_acc = val_student(student_model, dataloader['dev_loader'], device)
                best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

                # test on the test set
                test_loss, test_acc = val_student(student_model, dataloader['test_loader'], device)
                test_acc_list.append(test_acc)

                print(
                    'Test: epoch: {}, student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                        epoch, student_updates, train_loss, train_acc, val_loss, val_acc))
                print('Test: epoch{}, student_iter: {}, test loss: {:.4f}, test acc: {:.4f}'.format(epoch, student_updates, test_loss,
                                                                                           test_acc))
                writer.add_scalars('test_l2t/train', {'train_loss': train_loss, 'train_acc': train_acc},
                                   student_updates)
                writer.add_scalars('test_l2t/val', {'val_loss': val_loss, 'val_acc': val_acc}, student_updates)
                writer.add_scalars('test_l2t/test', {'test_loss': test_loss, 'test_acc': test_acc}, num_effective_data)

                if num_effective_data >= config['num_effective_data']:
                    return

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to teach')
    parser.add_argument('--tau', type=float, default=0.80)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_episode', type=int, default=300)
    parser.add_argument('--traj_num_per_episode', type=int, default=8)
    parser.add_argument('--epoch_num_per_traj', type=int, default=20)
    parser.add_argument('--lr_increment', type=float, default=0.0002)
    parser.add_argument('--train_episode', type=int, default=300)
    parser.add_argument('--test_epoch', type=int, default=540)
    parser.add_argument('--num_effective_data', type=int, default=4500000)
    args = parser.parse_args()

    # set seed
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'tau': args.tau,
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
        'num_effective_data': args.num_effective_data,
        'path_to_dataset': './data',
        'tensorboard_save_path': './runs/l2t_cifar10',
        'teacher_save_dir': './result/l2t/teacher',
        'teacher_save_model': 'teacher_step1_cifar10.pth',
        'student_save_dir': './result/l2t/student',
        'student_save_model': 'student_step2_cifar10.pth'
    }
    teacher_model = teacher_mlp().to(device)
    student_model = resnet32().to(device)

    for name, param in student_model.named_parameters():
        print(name)

    input()

    layer_list = []  # TODO: add in contents

    lr_config_list = create_configs_predefined(layer_list, lr_list=[0.001] * len(layer_list))

    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset=config['path_to_dataset'])

    writer = SummaryWriter(config['tensorboard_save_path'])

    print('Training the teacher starts....................')
    start = time.time()
    train_l2t()
    time_train = time.time() - start

    print('Saving the teahcer model........................')
    model_save_dir = config['teacher_save_dir']
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(teacher_model.state_dict(), os.path.join(model_save_dir, config['teacher_save_model']))

    print('Done.\nTesting the teacher.....................')
    print('Loading the teacher model')
    teacher_model.load_state_dict(torch.load(os.path.join(model_save_dir, config['teacher_save_model'])))
    start = time.time()
    test_l2t()
    time_test = time.time() - start

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_train//3600, time_train // 60, time_train % 60))
    print('Testing complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_test // 3600, time_test // 60, time_test % 60))

    print('Saving the student mdoel.......................')
    model_save_dir = './result/reinforce_cifar10/student'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(student_model.state_dict(), os.path.join(model_save_dir, config['student_save_model']))
