# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Simple transformer model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed

import sys
sys.path.append('/home/wuhaolu/Documents/pose_prediction/')
from act.utils import INPUT_DIM


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class SimpleTransformer(nn.Module):
    
    def __init__(self, transformer, encoder, state_dim, num_queries):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        history_state = 30
        self.input_proj_robot_state = nn.Linear(INPUT_DIM * 2 * history_state, hidden_dim)
        self.input_proj_slam = nn.Linear(INPUT_DIM, hidden_dim)
        self.input_proj_phase1 = nn.Linear(INPUT_DIM, hidden_dim)
        self.input_proj_env_state = nn.Linear(INPUT_DIM, hidden_dim)
        self.pos = torch.nn.Embedding(1, hidden_dim)
        self.backbones = None
        
        # encoder extra params
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(INPUT_DIM, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(INPUT_DIM * 2 * history_state, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pose_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        # print("qpos", qpos.shape)
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)

            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)

            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
        else:
            # print("Inference")
            mu = logvar = None
        
        qpos = self.input_proj_robot_state(qpos)
        qpos = torch.unsqueeze(qpos, axis=1)  # (bs, 1, hidden_dim)
        
        # env_state = self.input_proj_env_state(env_state)
        transformer_input = torch.cat([qpos], axis=1) # seq length = 1
        hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]