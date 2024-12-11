import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data, load_data_euroc, load_test_euroc # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, SimplePolicy
from visualize_episodes import save_videos

import csv
import time
import pandas as pd

import sys
sys.path.append('/home/wuhaolu/Documents/pose_prediction/PosePrediction')
from utils import * 
from predictPoseAct import generateTrainIndex

from sim_env import BOX_POSE

import IPython
e = IPython.embed

prediction_window = [0, 
                     10, 
                     18, 
                    #  45, 
                    #  90
                     ]



def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = INPUT_DIM
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': args['chunk_size'],
                         'camera_names': camera_names,}
    elif policy_class == 'SIMPLE':
        enc_layers = 1
        dec_layers = 2
        nheads = 4
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }

    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            # success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            eval_bc_euroc(config, ckpt_name)
            # results.append([ckpt_name, success_rate, avg_return])

        # for ckpt_name, success_rate, avg_return in results:
        #     print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        # exit()
        return
    # train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # # save dataset stats
    # if not os.path.isdir(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    # with open(stats_path, 'wb') as f:
    #     pickle.dump(stats, f)
    print("Batch size in train is ", batch_size_train)
    train_dataloader, val_dataloader, _, _ = load_data_euroc(num_episodes, batch_size_train, batch_size_val, policy_class)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

    # Writeback the params and other key factors for this training
    with open("model_param.txt", "w") as param_file:
        param_file.write("model name: "+ policy_class + "\n")
        param_file.write("encoder layers: "+ str(policy_config['enc_layers']) + "\n")
        param_file.write("dec_layers: "+ str(policy_config['dec_layers']) + "\n")
        param_file.write("nheads: "+ str(policy_config['nheads']) + "\n")
        param_file.write("test_index: "+ str(TEST_IDX) + "\n")
        param_file.write("test_name: "+ MSD_LIST[TEST_IDX] + "\n")



def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'SIMPLE':
        policy = SimplePolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'SIMPLE':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc_euroc(config, ckpt_name):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    # temporal_agg = config['temporal_agg']
    temporal_agg = False
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # get the dataset
    dataset = load_test_euroc(TEST_IDX, policy_class)

    groundtruth = dataset.getGroundtruth()
    slam_output = dataset.getSlamSource()

    gt_length = len(groundtruth)

    num_queries = policy_config['num_queries']

    all_time_actions = torch.zeros([num_queries, num_queries, state_dim]).cpu()

    with torch.inference_mode():
        
        fig, ax = plt.subplots()

        print(gt_length)
        # print("num queries:", num_queries)

        
        action_windows = [[] for _ in range(len(prediction_window))]

        batch_size = 20

        if policy_class == 'SIMPLE':
            begin_idx = 30
        else:
            begin_idx = 0
        print(gt_length // 2)

        for t in range(begin_idx, gt_length // 2, batch_size):
        # for t in range(begin_idx, 102, batch_size):
            print(t)
            
            if policy_class == 'SIMPLE':
                qpos = dataset.getPoseAt(t, batch_size)

                time_begin = time.time_ns()
                all_actions = policy(qpos)
                time_end = time.time_ns()
                print("Inference time:", (time_end - time_begin) / 1000000, " ms")
            else:

                image, qpos = dataset.getImagePoseAt(t, batch_size)
                
                time_begin = time.time_ns()
                all_actions = policy(qpos, image)
                time_end = time.time_ns()
                print("Inference time:", (time_end - time_begin) / 1000000, " ms")
            
            print(all_actions.shape)
            if not temporal_agg: 
                # We aim at the target of different prediction window
                for i, window in enumerate(prediction_window):
                    raw_action = all_actions[:, window].squeeze(0).cpu().numpy()
                    # print(raw_action.shape)
                    action_windows[i].extend(raw_action)
                    # print(action_windows[i])
            else:
                
                for i, window in enumerate(prediction_window):
                    # Current we are targeting at prediction for t + window
                    # So we can at most use [t + window - num_queries] for doing prediction
                    print(all_time_actions.shape)
                    all_time_actions = all_time_actions[1:]

                    all_time_actions = torch.cat([all_time_actions, all_actions.cpu()], axis = 0)
                    print(all_time_actions.shape)

                    index = window
                    actions_for_curr_step = np.asarray([all_time_actions[len(all_time_actions)-1][index].numpy()])
                    print(actions_for_curr_step.shape)
                    
                    for j in range(len(all_time_actions)-2,-1,-1):
                        if index == 100 or all_time_actions[j][index][0] == 0.000 :
                            break
                        actions_for_curr_step = np.concatenate([actions_for_curr_step, [all_time_actions[j][index].numpy()]], axis = 0)
                        index += 1
                    
                    print(actions_for_curr_step.shape)
                    
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))).reshape((len(actions_for_curr_step), 1))
                    print(exp_weights)
                    exp_weights = exp_weights / exp_weights.sum()
                    # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)
                    # print("raw_action:", raw_action)
               
                    # action = raw_action[:, window].squeeze(0).cpu().numpy()
                    action_windows[i].append(raw_action)

                
        # Dump the current result 
        prefix = config['ckpt_dir']
        print("prefix: ", prefix)

        if INPUT_DIM == 6:
            header = "x, y, z, y, p, r\n"
        elif INPUT_DIM == 7:
            header = "x, y, z, w, x, y, z\n"
        else:
            header = "x, y, z, ysin, ycos, psin, pcos, rsin, rcos\n"
            

        for window, action_window in zip(prediction_window, action_windows):
            with open(prefix + '/' + "res_" + str(window) + ".csv", "w") as res:
                res.write(header)
                # res.write("x, y, z, yaw, pitch, row")
                csv_writer = csv.writer(res)
                try:
                    csv_writer.writerows(action_window)
                except:
                    print("window error:", action_window)

        # Compute the pose error rate
        for i, window in enumerate(prediction_window):
            raw_actions = action_windows[i]
            print(raw_actions)
            print(groundtruth)
            pose_diff_list, angle_diff_list = computePoseDiffFromNumpy(raw_actions[:-window], groundtruth[window:])

            data_sorted = np.sort(pose_diff_list)
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            ax.plot(data_sorted, cdf, label="Phase2 error rate with window " + str(window))

        # Draw the source errors
        pose_diff_computed_slam_and_gt_slam, \
        angle_diff_computed_slam_and_gt_slam = computePoseDiffFromNumpy(slam_output, groundtruth)
        
        data_sorted = np.sort(pose_diff_computed_slam_and_gt_slam)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax.plot(data_sorted, cdf, label="SLAM source error rate")

        ax.legend()

        fig.savefig("res_act.png")


def compute_result_from_euler_file(args):

    policy_class = args['policy_class']

    # Read the groundtruth
    dataset = load_test_euroc(TEST_IDX, policy_class)
    groundtruth = dataset.getGroundtruth()
    slam_output = dataset.getSlamSource()

    fig_pos, ax_pos = plt.subplots()
    fig_orient, ax_orient = plt.subplots(3)
    
    prefix = args['ckpt_dir'] + '/'
    print(prefix)

    for window in prediction_window:
        actions = pd.read_csv(prefix + "res_" + str(window) + ".csv").to_numpy()[:,:6]
        print("actions: ", len(actions))

        groundtruth = groundtruth[window: min(len(actions) + window, len(groundtruth))]
        actions = actions[:len(groundtruth)]

        print(groundtruth[len(groundtruth)//2 -2 ])
        print(actions[len(groundtruth)//2 - 2])

        print(actions.shape, groundtruth.shape)
        pose_diff_list, angle_diff_list = computePoseDiffFromNumpy(actions, groundtruth)

        yaw_list = angle_diff_list[:, 0]
        pitch_list = angle_diff_list[:, 1]
        roll_list = angle_diff_list[:, 2]

        print("windows size:", window)
        print(np.average(pose_diff_list), np.average(yaw_list), np.average(pitch_list), np.average(roll_list))
        
        data_sorted = np.sort(pose_diff_list)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax_pos.plot(data_sorted, cdf, label="Phase2 error rate with window " + str(window))

        data_sorted = np.sort(yaw_list)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax_orient[0].plot(data_sorted, cdf, label="Yaw error rate with window " + str(window))

        data_sorted = np.sort(pitch_list)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax_orient[1].plot(data_sorted, cdf, label="Pitch error rate with window " + str(window))

        data_sorted = np.sort(roll_list)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        ax_orient[2].plot(data_sorted, cdf, label="Roll error rate with window " + str(window))

    pose_diff_computed_slam_and_gt_slam, \
    angle_diff_computed_slam_and_gt_slam = computePoseDiffFromNumpy(slam_output, dataset.getGroundtruth())

    print("SLAM source average error:", 
          np.average(pose_diff_computed_slam_and_gt_slam),
          np.average(angle_diff_computed_slam_and_gt_slam, axis=0))

    data_sorted = np.sort(angle_diff_computed_slam_and_gt_slam[:,0])
    print(len(data_sorted))
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax_orient[0].plot(data_sorted, cdf, label="SLAM source Yaw error rate ")

    data_sorted = np.sort(angle_diff_computed_slam_and_gt_slam[:,1])
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax_orient[1].plot(data_sorted, cdf, label="SLAM source Pitch error rate")

    data_sorted = np.sort(angle_diff_computed_slam_and_gt_slam[:,2])
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax_orient[2].plot(data_sorted, cdf, label="SLAM source Roll error rate")
    
    data_sorted = np.sort(pose_diff_computed_slam_and_gt_slam)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax_pos.plot(data_sorted, cdf, label="SLAM source error rate")

    ax_pos.legend()
    ax_orient[0].legend()
    ax_orient[1].legend()
    ax_orient[2].legend()

    fig_pos.savefig("res_act_position.png")
    fig_orient.savefig("res_act_orient.png")


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
    avg_return = np.mean(episode_returns)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))


def forward_pass(data, policy, policy_class):
    if policy_class == 'SIMPLE':
        qpos_data, action_data, is_pad = data
        qpos_data, action_data, is_pad = qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, action_data, is_pad)
    else:
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, policy_class)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            # print("batch_idx, ", batch_idx)
            forward_dict = forward_pass(data, policy, policy_class)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print("")
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)




    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    generateTrainIndex()
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))

    compute_result_from_euler_file(vars(parser.parse_args()))