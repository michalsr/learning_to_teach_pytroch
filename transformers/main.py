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
from tensorboardX import SummaryWriter
from tqdm import trange
import time
import numpy as np
import logging 
np.set_printoptions(threshold=np.inf)


class L2T(object):
    def __init__(self,config,teacher_model,student_model,data_loader,writer,logger,device) -> None:
        self.config = config
        self.teacher_model= teacher_model 
        self.student_model = student_model 
        self.data_loader = data_loader 
        self.writer = writer 
        self.logger = logger 
        self.device = device
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

    def select_action(self,state):
        action_prb = self.teacher_model(state.detach())
        m = Categorical(action_prb)
        action = m.sample()
        self.teacher_model.saved_log_probs.append(m.log_prob(action))
        return action
    def student_lr_scheduler_cifar10(self,optimizer, iterations):
        lr = 0.0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        if iterations == 32000:
            lr = lr * 0.1
            self.logger.log(f'Adjust learning rate to {lr}')
           
        elif iterations == 48000:
            lr = lr * 0.1
            self.logger.log(f'Adjust learning rate to {lr}')
  
        else:
            return optimizer 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer 
    def val_student(self,model, dataloader, device):
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
    def train_l2t(self):
        num_steps_to_achieve = []
        non_increasing_steps = 0
        for i_episode in range(self.config['train_episode']):
            training_loss_history = []
            best_loss_on_dev = self.config['max_loss']
            student_updates = 0
            i_iter = 0
            input_pool = []
            label_pool = []
            done = False
            count_sampled = 0

            # init the student
            self.student_model.apply(_weights_init)

            # one episode
            while True:

                for idx, (inputs, labels) in enumerate(self.dataloader['teacher_train_loader']):

                    # compute the state feature
                    state_config = {
                        'inputs': inputs.to(self.device),
                        'labels': labels.to(self.device),
                        'num_class': self.config['num_classes'],
                        'student_iter': student_updates,
                        'training_loss_history': training_loss_history,
                        'best_loss_on_dev': best_loss_on_dev,
                        'model': self.student_model,
                        'max_loss': self.config['max_loss'],
                        'max_iter': self.config['max_iter'],
                        'device': device
                    }
                    state = self.state_func(state_config)

                    # the teacher select action according to the state feature
                    action = self.select_action(state)

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
                            'model': self.student_model,
                            'device': device,
                            'optimizer': optim.SGD(self.student_model.parameters(), lr=0.1, momentum=0.9,
                                                weight_decay=1e-4)
                        }

                        # train the student
                        train_loss, train_acc = self.train_student(train_student_config)
                        training_loss_history.append(train_loss)
                        student_updates += 1
                        optimizer = self.student_lr_scheduler_cifar10(self,train_student_config['optimizer'], student_updates)

                        # val the student on validation set
                        val_loss, val_acc = self.val_student(student_model, dataloader['dev_loader'], self.device)
                        best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev
                        self.logging.log(f'episode: {i_episode}, student_iter: {student_updates}, train loss: {train_loss}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}')
                        # print(
                        #     'episode:{}, student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                        #         i_episode, student_updates, train_loss, train_acc, val_loss, val_acc))
                        # # writer.add_scalars('train_l2t/train', {'train_loss': train_loss, 'train_acc': train_acc},
                        #                    global_step=student_updates)
                        # writer.add_scalars('train_l2t/val', {'val_loss': val_loss, 'val_acc': val_acc}, global_step=student_updates)

                        if val_acc >= config['tau'] or i_iter == config['max_iter']:
                            num_steps_to_achieve.append(i_iter)
                            reward_T = -math.log(i_iter / config['max_iter'])
                            teacher_model.rewards.append(reward_T)
                            teacher_model.reward_T_histtory.append(reward_T)
                            done = True
                            print('acc >= 0.80 at {}， reward_T: {}'.format(i_iter, reward_T))
                            print('=' * 30)
                            print(teacher_model.reward_T_histtory)
                            print('=' * 30)
                            writer.add_scalar('num_to_achieve', i_iter, i_episode)
                            writer.add_scalar('reward', reward_T, i_episode)
                            break
                        else:
                            teacher_model.rewards.append(0)

                if done == True:
                    break

            # update teacher
            update_teacher_config = {
                'non_increasing_steps': non_increasing_steps,
                'i_episode': i_episode,
                'batch_size': config['batch_size'],
                'optimizer': optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=0)
            }
            non_increasing_steps = update_teacher(update_teacher_config)

            writer.add_scalar('non_increasing_steps', non_increasing_steps)
            teacher_model.rewards = []
            teacher_model.saved_log_probs = []

            if non_increasing_steps >= config['max_non_increasing_steps']:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(teacher_model.reward_T_histtory[-1], i_episode))
                return
    def state_func(self,state_config):
        inputs = state_config['inputs']
        labels = state_config['labels']
        n_samples = inputs.size(0)
        student_model = state_config['model']
        max_loss = state_config['max_loss']
        best_loss_on_dev = state_config['best_loss_on_dev']

        student_model.eval()
        outputs = student_model(inputs)
        outputs = nn.Softmax(dim=1)(outputs)

        average_train_loss = max_loss if len(state_config['training_loss_history']) == 0 else sum(
            state_config['training_loss_history']) / len(state_config['training_loss_history'])
        log_P_y = torch.log(outputs[range(n_samples), labels.data]).reshape(-1, 1)
        mask = torch.ones(inputs.size(0), state_config['num_class']).to(state_config['device'])
        mask[range(n_samples), labels.data] = 0
        margin_value = (outputs[range(n_samples), labels.data] - torch.max(mask * outputs, 1)[0]).reshape(-1, 1)

        state_features = torch.zeros((inputs.size(0), 25)).to(self.device)
        state_features[range(n_samples), labels.data] = 1
        state_features[range(n_samples), 10] = state_config['student_iter'] / state_config['max_iter']
        self.logging.log(f'Average train loss: {average_train_loss}')
        state_features[range(n_samples), 11] = average_train_loss / max_loss
        state_features[range(n_samples), 12] = best_loss_on_dev / max_loss
        state_features[range(n_samples), 13:23] = outputs
        state_features[range(n_samples), 23] = torch.from_numpy(
            rankdata(-log_P_y.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)
        state_features[range(n_samples), 24] = torch.from_numpy(
            rankdata(margin_value.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)

        return state_features


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to teach')
    parser.add_argument('--tau', type=float, default=0.80)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_episode', type=int, default=300)
    parser.add_argument('--test_epoch', type=int, default=540)
    parser.add_argument('--num_effective_data', type=int, default=4500000)
    parser.add_argument('--output_dir',type=str)
    parser.add_argument('--file_prefix',type=str)
    args = parser.parse_args()

    # set seed
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'tau': args.tau,
        'max_iter': 10000,
        'batch_size': args.batch_size,
        'max_non_increasing_steps': 5,
        'num_classes': 10,
        'max_loss': 15,
        'train_episode': args.train_episode,
        'test_episode': args.test_epoch,
        'num_effective_data': args.num_effective_data,
        'path_to_dataset': f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}/data',
        'tensorboard_save_path': f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}/runs/l2t_cifar10',
        'teacher_save_dir': f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}/result/l2t/teacher',
        'teacher_save_model': f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}/teacher_step1_cifar10.pth',
        'student_save_dir': f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}/result/l2t/student',
        'student_save_model': f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}student_step2_cifar10.pth'
    }
   
    teacher_model = teacher_mlp().to(device)
    student_model = resnet32().to(device)

    dataloader = data_loader_func(batch_sizes=config['batch_size'], path_to_dataset=config['path_to_dataset'])

    writer = SummaryWriter(config['tensorboard_save_path'])
    logger = setup_logger('logger', args.output_dir+'/log_file.log')

    print('Training the teacher starts....................')
    l2t = L2T(config=config,teacher_model=teacher_model,student_model=student_model,data_loader=dataloader,writer=writer,logger=logger)
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
