import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch

import sys
sys.path.append('/home/wuhaolu/Documents/pose_prediction/')
from PosePrediction.utils import *

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_simple_transformer
import IPython
e = IPython.embed

def custom_loss(y_pred, y_true):

    # Example: Mean Absolute Error (MAE)

    penalty = torch.sum((y_pred[:,3] < -1).float()) + torch.sum((y_pred[:,3] > 1).float())
    penalty += torch.sum((y_pred[:,4] < -1).float()) + torch.sum((y_pred[:,4] > 1).float())
    penalty += torch.sum((y_pred[:,5] < -1).float()) + torch.sum((y_pred[:,5] > 1).float())
    # print("penalty: ", penalty)

    loss = torch.mean(torch.abs(y_pred[:,0] - y_true[:,0]))
    # print(y_pred, y_true)
    loss += torch.mean(torch.abs(y_pred[:,1] - y_true[:,1]))
    loss += torch.mean(torch.abs(y_pred[:,2] - y_true[:,2]))
    loss += torch.mean(torch.abs((y_pred[:,3] - y_true[:,3] + 1) % 2 -1))
    loss += torch.mean(torch.abs((y_pred[:,4] - y_true[:,4] + 1) % 2 -1))
    loss += torch.mean(torch.abs((y_pred[:,5] - y_true[:,5] + 1) % 2 -1))

    loss = loss / 6.0
    return loss + penalty / 20.0
    # return loss + penalty

def custom_loss_quat(y_pred, y_true):
    loss_pose = F.mse_loss(y_pred[:,:3], y_true[:, :3])
    loss_quat = F.mse_loss(y_pred[:,3:7], y_true[:, 3:7])
    beta = 10
    return loss_pose + beta * loss_quat



class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = custom_loss_quat(actions, a_hat)
            # pose_diff, angle_diff = computePoseDiffFromNumpy(actions.cpu().detach().numpy()[0], 
            #                                                  a_hat.cpu().detach().numpy()[0])
            # poseLoss = CustomLoss()
            # pose_diff, angle_diff = poseLoss(actions, a_hat)

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld




class SimplePolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_simple_transformer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.input_size = 30
        print(f'KL Weight {self.kl_weight}')
    
    def __call__(self, qpos, actions=None, is_pad=None):
        
        if actions is not None:

            actions = actions[:, :self.model.num_queries]
            # print("actionshape: ", actions.shape)
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = custom_loss_quat(actions, a_hat)
            
            # print("a_hat:", a_hat)
            # hat, _, _  = self.model(qpos)
            # print("hat: ", hat)

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            # loss_dict['pose'] = torch.tensor((np.average(pose_diff) + np.average(angle_diff)) / 2).cuda(0)
            # print("pose loss is ", loss_dict['pose'], loss_dict['l1'])

            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos) # no action, sample from prior
            return a_hat
    
    def configure_optimizers(self):
        return self.optimizer