import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from network.teacher_model.teacher_mlp import teacher_mlp
from network.student_model.resnet import resnet32, _weights_init
from dataset.data_loader import data_loader_func_no_train_transform,apply_transform
from utils.setseed import setup_seed
from scipy.stats import rankdata
import math
import argparse
import copy 
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import numpy as np
import logging 
from dataset.aug_dict import make_aug_dict
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
        
    def teacher_lr_scheduler(self,optimizer, iterations):
        lr = 0.0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        if iterations % 50 == 0:
            lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer 
    def update_teacher(self,config):
        optimizer = config['optimizer']
        i_episode = config['i_episode']
        rewards = self.teacher_model.rewards
        saved_log_probs = self.teacher_model.saved_log_probs
        reward_T_histtory = self.teacher_model.reward_T_histtory

        optimizer.zero_grad()

        reward_T = rewards[-1]
        self.teacher_model.rewards_baseline = self.teacher_model.rewards_baseline + (reward_T - self.teacher_model.rewards_baseline) / (i_episode + 1)
        baseline = 0.0 if i_episode == 0 else self.teacher_model.rewards_baseline
        policy_loss = -torch.cat(saved_log_probs).sum() * (reward_T - baseline)
        policy_loss = policy_loss / config['batch_size']
        policy_loss.backward()
        optimizer.step()
        self.logger.info(f'policy loss for {i_episode} is {policy_loss}')
        self.logger.info(f'baseline for {i_episode} for {baseline}')

        self.writer.add_scalar('policy_loss', policy_loss, i_episode)
        self.writer.add_scalar('baseline', baseline, i_episode)

        optimizer = self.teacher_lr_scheduler(optimizer, i_episode)

        last_reward = 0.0 if len(reward_T_histtory) == 1 else reward_T_histtory[-2]
        if abs(last_reward - reward_T) < 0.01:
            config['non_increasing_steps'] += 1
        else:
            config['non_increasing_steps'] = 0

        return config['non_increasing_steps'],optimizer 
    def train_student(self,config):
        model = config['model']
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        #inputs = config['inputs'].to(config['device'])
        labels = config['labels'].to(config['device'])

        # zero the parameter gradients
        optimizer = config['optimizer']
        optimizer.zero_grad()
        augmentations = self.aug_dict[config['aug']]
        inputs = apply_transform(config['inputs'],augmentations)
        inputs = inputs.to(config['device'])
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

        return train_loss, train_acc, model

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
            self.logger.info(f'Adjust learning rate to {lr}')
           
        elif iterations == 48000:
            lr = lr * 0.1
            self.logger.info(f'Adjust learning rate to {lr}')
  
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
        teacher_opt = optim.Adam(self.teacher_model.parameters(), lr=0.001, weight_decay=0)
        #vis = tqdm(range(self.config['train_episode']))
        for i_episode in tqdm(range(self.config['train_episode'])):
            self.logger.info(f'Epsidoe {i_episode}')
            training_loss_history = []
            best_loss_on_dev = self.config['max_loss']
            student_updates = 0
            i_iter = 0
            input_pool = []
            label_pool = []
            done = False
            count_sampled = 0
            episode_student_model = copy.deepcopy(self.student_model)
            episode_student_model.apply(_weights_init)
            episode_student_optimizer = optim.SGD(episode_student_model.parameters(), lr=0.1, momentum=0.9,
                                                weight_decay=1e-4)

       

            # one episode
            while True:

                for idx, (inputs, labels) in enumerate(self.data_loader['teacher_train_loader']):
                    # each mini batch is a trajectory 
                    # compute the state feature
                    teacher = len(self.data_loader['teacher_train_loader'])
                    self.logger.info(f'On index {idx} of {teacher}')
                    state_config = {
                        'inputs': inputs.to(self.device),
                        'labels': labels.to(self.device),
                        'num_class': self.config['num_classes'],
                        'student_iter': student_updates,
                        'training_loss_history': training_loss_history,
                        'best_loss_on_dev': best_loss_on_dev,
                        'model': episode_student_model,
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
                    indices = torch.nonzero(action)[0]
                    train_student_config = {
                            'inputs': inputs,
                            'labels': labels,
                            'model': episode_student_model,
                            'aug':indices.item(),
                            'device': device,
                            'optimizer': episode_student_optimizer
                        }

                    # train the student
                    train_loss, train_acc,episode_student_model = self.train_student(train_student_config)
                    training_loss_history.append(train_loss)
                    student_updates += 1
                    episode_student_optimizer = self.student_lr_scheduler_cifar10(episode_student_optimizer, student_updates)

                    # val the student on validation set
                    val_loss, val_acc = self.val_student(episode_student_model, self.data_loader['dev_loader'], self.device)
                    best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev
                    self.logger.info(f'episode: {i_episode}, student_iter: {student_updates}, train loss: {train_loss}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}')
                    if val_acc >= self.config['tau'] or i_iter == self.config['max_iter']:
                        num_steps_to_achieve.append(i_iter)
                        reward_T = -math.log(i_iter / config['max_iter'])
                        self.teacher_model.rewards.append(reward_T)
                        self.teacher_model.reward_T_histtory.append(reward_T)
                        done = True
                        self.logger.info(f'acc >= 0.80 at {i_iter}， reward_T: {reward_T}')
                        self.logger.info(f'iteration: {i_iter}, episode: {i_episode}')
                        self.logger.info(f'reward {reward_T} for episode {i_episode}')
                    
                        self.writer.add_scalar('num_to_achieve', i_iter, i_episode)
                        self.writer.add_scalar('reward', reward_T, i_episode)
                        break
                    else:
                        self.teacher_model.rewards.append(0)

                if done == True:
                    break
                    # if len(indices) == 0:
                    #     continue

                    # count_sampled += len(indices)
                    # selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                    # selected_labels = labels[indices.squeeze()].view(-1, 1)
                    # input_pool.append(selected_inputs)
                    # label_pool.append(selected_labels)

                    # if count_sampled >= config['batch_size']:
                    #     inputs = torch.cat(input_pool, 0)[:config['batch_size']]
                    #     labels = torch.cat(label_pool, 0)[:config['batch_size']].squeeze()

                    #     input_pool = []
                    #     label_pool = []
                    #     count_sampled = 0

                        
                        # print(
                        #     'episode:{}, student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                        #         i_episode, student_updates, train_loss, train_acc, val_loss, val_acc))
                        # # writer.add_scalars('train_l2t/train', {'train_loss': train_loss, 'train_acc': train_acc},
                        #                    global_step=student_updates)
                        # writer.add_scalars('train_l2t/val', {'val_loss': val_loss, 'val_acc': val_acc}, global_step=student_updates)

                       

            # update teacher
            update_teacher_config = {
                'non_increasing_steps': non_increasing_steps,
                'i_episode': i_episode,
                'batch_size': self.config['batch_size'],
                'optimizer': teacher_opt 
            }
            non_increasing_steps,teacher_opt = self.update_teacher(update_teacher_config)

            self.writer.add_scalar('non_increasing_steps', non_increasing_steps)
            self.teacher_model.rewards = []
            self.teacher_model.saved_log_probs = []

            if non_increasing_steps >= self.config['max_non_increasing_steps']:
                self.logger.info(f"Solved! Running reward is now {self.teacher_model.reward_T_histtory[-1]} and the last episode runs to {i_episode} steps")
               
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
        # log_P_y = torch.log(outputs[range(n_samples), labels.data]).reshape(-1, 1)
        # mask = torch.ones(inputs.size(0), state_config['num_class']).to(state_config['device'])
        # mask[range(n_samples), labels.data] = 0
        # margin_value = (outputs[range(n_samples), labels.data] - torch.max(mask * outputs, 1)[0]).reshape(-1, 1)

        state_features = torch.zeros((len(self.aug_dict), 4)).to(self.device)
        state_features[range(len(self.aug_dict)),0] = state_config['student_iter'] / state_config['max_iter']
        state_features[range(len(self.aug_dict)),1] = average_train_loss / max_loss
        state_features[range(len(self.aug_dict)),2] =  best_loss_on_dev / max_loss
        state_features[range(len(self.aug_dict)),3] = torch.from_numpy(
            rankdata(np.arange(len(self.aug_dict)))/len(self.aug_dict)).to(state_config['device'], dtype=torch.float)

        # state_features[range(n_samples), labels.data] = 1
        # state_features[range(n_samples), 10] = state_config['student_iter'] / state_config['max_iter']
        # self.logger.info(f'Average train loss: {average_train_loss}')
        # state_features[range(n_samples), 11] = average_train_loss / max_loss
        # state_features[range(n_samples), 12] = best_loss_on_dev / max_loss
        # state_features[range(n_samples), 13:23] = outputs
        # state_features[range(n_samples), 23] = torch.from_numpy(
        #     rankdata(-log_P_y.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)
        # state_features[range(n_samples), 24] = torch.from_numpy(
        #     rankdata(margin_value.detach().cpu().numpy()) / n_samples).to(state_config['device'], dtype=torch.float)

        return state_features
    def test(self):
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
        test_student_model = copy.deepcopy(self.student_model)
        test_student_model.apply(_weights_init)
        test_student_optimizer = optim.SGD(test_student_model.parameters(), lr=0.1, momentum=0.9,
                                            weight_decay=0.0001)
     

        for epoch in range(self.config['test_episode']):
            for idx, (inputs, labels) in enumerate(self.data_loader['student_train_loader']):

                # compute the state feature
                state_config = {
                    'inputs': inputs.to(device),
                    'labels': labels.to(device),
                    'num_class': self.config['num_classes'],
                    'student_iter': student_updates,
                    'training_loss_history': training_loss_history,
                    'best_loss_on_dev': best_loss_on_dev,
                    'model': test_student_model,
                    'max_loss': self.config['max_loss'],
                    'max_iter': self.config['max_iter'],
                    'device': self.device
                }
                state = self.state_func(state_config)

                # the teacher select action according to the state feature
                action = self.select_action(state)

                # finish one step
                i_iter += 1

                # collect data to train student
                indices = torch.nonzero(action)[0]
                train_student_config = {
                    'inputs': inputs,
                    'labels': labels,
                    'model': test_student_model,
                    'aug':indices.item(),
                    'device': self.device,
                    'optimizer': test_student_optimizer
                }
                train_loss, train_acc,test_student_model = self.train_student(train_student_config)
                training_loss_history.append(train_loss)
                student_updates += 1
                num_effective_data += inputs.size(0)
                test_student_optimizer = self.student_lr_scheduler_cifar10(train_student_config['optimizer'], student_updates)

                # val the student on validation set
                val_loss, val_acc = self.val_student(test_student_model, dataloader['dev_loader'], self.device)
                best_loss_on_dev = val_loss if val_loss < best_loss_on_dev else best_loss_on_dev

                # test on the test set
                test_loss, test_acc = self.val_student(test_student_model, dataloader['test_loader'], self.device)
                test_acc_list.append(test_acc)
                self.logger.info(f'Test: epoch: {epoch}, student_iter:{student_updates},train loss: {train_loss}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}')
                self.logger.info(f'Test: epoch {epoch}, student_iter:{student_updates}, test loss: {test_loss}, test acc: {test_acc}')
                self.writer.add_scalars('test_l2t/train', {'train_loss': train_loss, 'train_acc': train_acc},
                                    student_updates)
                self.writer.add_scalars('test_l2t/val', {'val_loss': val_loss, 'val_acc': val_acc}, student_updates)
                self.writer.add_scalars('test_l2t/test', {'test_loss': test_loss, 'test_acc': test_acc}, num_effective_data)

                    # if num_effective_data >= config['num_effective_data']:
                    #     return

                # if len(indices) == 0:
                #     continue

                # count_sampled += len(indices)
                # selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                # selected_labels = labels[indices.squeeze()].view(-1, 1)
                # input_pool.append(selected_inputs)
                # label_pool.append(selected_labels)

                # if count_sampled >= self.config['batch_size']:
                #     inputs = torch.cat(input_pool, 0)[:self.config['batch_size']]
                #     labels = torch.cat(label_pool, 0)[:self.config['batch_size']].squeeze()

                #     input_pool = []
                #     label_pool = []
                #     count_sampled = 0

               
                # train the student
                    # print(
                    #     'Test: epoch: {}, student_iter: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(
                    #         epoch, student_updates, train_loss, train_acc, val_loss, val_acc))
                    # print('Test: epoch{}, student_iter: {}, test loss: {:.4f}, test acc: {:.4f}'.format(epoch, student_updates, test_loss,
                    #                                                                         test_acc))
                   
        return 



formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO,resume=False):
    """To setup as many loggers as you want"""
    if not resume:
        os.remove(log_file)

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    handler_2 = logging.StreamHandler()
    logger.addHandler(handler_2)

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to teach')
    parser.add_argument('--tau', type=float, default=0.80)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
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
        'path_to_dataset': f'{args.file_prefix}/learning_to_teach_pytroch/l2t_aug/data',
        'tensorboard_save_path': f'{args.file_prefix}/learning_to_teach_pytroch/l2t_aug/{args.output_dir}/runs/l2t_cifar10',
        'teacher_save_dir': f'{args.file_prefix}/learning_to_teach_pytroch/l2t_aug/{args.output_dir}/result/l2t/teacher',
        'teacher_save_model': f'{args.file_prefix}/learning_to_teach_pytroch/l2t_aug/{args.output_dir}/teacher_step1_cifar10.pth',
        'student_save_dir': f'{args.file_prefix}/learning_to_teach_pytroch/l2t_aug/{args.output_dir}/result/l2t/student',
        'student_save_model': f'{args.file_prefix}/learning_to_teach_pytroch/l2t_aug/{args.output_dir}student_step2_cifar10.pth'
    }
    #student_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes = config['num_classes'],img_size=32)
    student_model = resnet32().to(device)
    teacher_model = teacher_mlp().to(device)
    #student_model = resnet32().to(device)

    dataloader = data_loader_func_no_train_transform(batch_sizes=config['batch_size'], path_to_dataset=config['path_to_dataset'])

    writer = SummaryWriter(config['tensorboard_save_path'])
    logger = setup_logger('logger', args.output_dir+'/log_file.log')

    logger.info('Training the teacher starts....................')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2t = L2T(config=config,teacher_model=teacher_model,student_model=student_model,data_loader=dataloader,writer=writer,logger=logger,device=device)
    l2t.aug_dict = make_aug_dict()
    start = time.time()
    l2t.train_l2t()
    time_train = time.time() - start

    logger.info('Saving the teahcer model........................')
    model_save_dir = config['teacher_save_dir']
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(teacher_model.state_dict(), os.path.join(model_save_dir, config['teacher_save_model']))

    logger.info('Done.\nTesting the teacher.....................')
    logger.info('Loading the teacher model')
    teacher_model.load_state_dict(torch.load(os.path.join(model_save_dir, config['teacher_save_model'])))
    start = time.time()
    l2t.test_l2t()
    time_test = time.time() - start
    logger.info(f'Training complete in {time_train//3600}, {time_train//60},{time_train % 60}')
    logger.info(f'Testing complete in {time_test//3600}, {time_test // 60}, {time_test % 60}')
    #print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_train//3600, time_train // 60, time_train % 60))
    #print('Testing complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_test // 3600, time_test // 60, time_test % 60))
    
    logger.info('Saving the student model.......................')
    model_save_dir = f'{args.file_prefix}/learning_to_teach_pytorch/transformers/{args.output_dir}/result/reinforce_cifar10/student'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(student_model.state_dict(), os.path.join(model_save_dir, config['student_save_model']))
