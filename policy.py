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


# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()

#     def forward(self, y_pred, y_true):
#         # Example: Combine L1 Loss and L2 Loss
#         def compute_pose_diff(y_pred, y_true):
#             print(y_pred, y_true)
#             position1 = torch(y_pred[0], y_pred[1], y_pred[2])
#             position2 = torch(y_true[0], y_true[1], y_true[2])

#             orientation1 = (y_pred[3], y_pred[4], y_pred[5], y_pred[6])
#             orientation2 = (y_true[3], y_true[4], y_true[5], y_true[6])
#             elucid_diff = torch.norm(position1 - position2)
        

#             rotation1 = Rotation.from_quat(orientation1)
#             rotation2 = Rotation.from_quat(orientation2)

#             rotated1 = rotation1.apply((0,0, 1))
#             rotated2 = rotation2.apply((0,0, 1))

#             angle_diff = np.arccos(np.dot(rotated1,rotated2) / 
#                                 (np.dot(rotated1, rotated1) * 
#                                     np.dot(rotated2, rotated2)))
            
#             return elucid_diff, angle_diff
        
#         return elucid_diff + angle_diff * 0.5

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
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            
            # pose_diff, angle_diff = computePoseDiffFromNumpy(actions.cpu().detach().numpy()[0], 
            #                                                  a_hat.cpu().detach().numpy()[0])
            # poseLoss = CustomLoss()
            # pose_diff, angle_diff = poseLoss(actions, a_hat)
            # print(pose_diff, angle_diff)

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            # loss_dict['pose'] = torch.tensor((np.average(pose_diff) + np.average(angle_diff)) / 2).cuda(0)
            # print("pose loss is ", loss_dict['pose'], loss_dict['l1'])

            return loss_dict
        else: # inference time
            # TODO: For the pose prediction, the model should also accept input for a future time
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
            actions = actions[:, 0]
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
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            
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
            for i in range(0, len(qpos), 7):
                print(qpos[i:i+7])
            print("|||||||")
            a_hat, _, (_, _) = self.model(qpos) # no action, sample from prior
            print("a_hat inference:", a_hat)
            return a_hat
    
    def configure_optimizers(self):
        return self.optimizer