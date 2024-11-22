import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('/home/wuhaolu/Documents/pose_prediction/')
from PosePrediction.Utils import * 

import pandas as pd
import cv2

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        
        # Load the dataset
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            
            # Get the start index from the sample
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            print("qos shape :", qpos.shape)
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        # Padding the action into the whole episode length
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        # print(image_data.shape)
        # print(qpos_data.shape)
        # print(action_data.shape)
        # print(is_pad.shape)
        # print("")
        return image_data, qpos_data, action_data, is_pad



class EuroCStyleDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, id2datasetpath, id2gtpath, id2camera_paths, ):

        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.id2datasetpath = id2datasetpath
        self.id2gtpath = id2gtpath
        self.id2camerapaths = id2camera_paths
        self.is_sim = None
    
    def __len__(self):
        return len(self.episode_ids)
    
    def __getitem__(self, index):

        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        
        # Get the path to groundtruth containing each action state's pose
        # And also, we should also include the current pose in the groundtruth poses
        # So we need to get each state's format

        # Load the data from prediction
        observation_path = self.id2datasetpath[episode_id]
        groundtruth_path = self.id2gtpath[episode_id]
        camera_path = self.id2camerapaths[episode_id]

        # Read the datasets from datasetpath with

        observation_raw = pd.read_csv(observation_path).to_numpy()
        # Get only the poses  
        slam_pose = observation_raw[:, 1:8]
        phase1_pose = observation_raw[:, 9:16]
        observation = np.hstack([slam_pose, phase1_pose])

        groundtruth_raw = pd.read_csv(groundtruth_path).to_numpy()
        groundtruth = groundtruth_raw[:, 1:]
        
        camera_path = pd.read_csv(camera_path).to_numpy()
        

        episode_len = observation.shape[0]
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        
        action_len = episode_len - start_ts 
        
        # For qos, we include the current slam observation & prediction
        qpos = observation[start_ts]
        left_image = np.asarray(cv2.imread(camera_path[start_ts][1]))
        right_image = np.asarray(cv2.imread(camera_path[start_ts][3]))

        # Load the image from information
        all_cam_images = [left_image, right_image]
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Construct action data
        # TODO: Check whether we also include the current action here
        action = groundtruth[start_ts:]

        original_action_shape = groundtruth.shape

        # print("shape:", original_action_shape, episode_len, observation.shape[0], groundtruth.shape[0])

        # Padding the action into the whole episode length
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # construct observations
        image_data = torch.from_numpy(all_cam_images) / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # TODO: Check whether we need to do normalization here
        # print(image_data.shape)
        # print(qpos_data.shape)
        # print(action_data.shape)
        # print(is_pad.shape)
        # print("")
        return image_data, qpos_data, action_data, is_pad
        # return image_data[:10000], qpos_data[:10000], action_data[:10000], is_pad[:10000]

    def getImagePoseAt(self, index, batch_size):

        episode_id = len(MSD_LIST)-1

        observation_path = self.id2datasetpath[episode_id]
        camera_path = self.id2camerapaths[episode_id]

        observation_raw = pd.read_csv(observation_path).to_numpy()
        # Get only the poses  
        slam_pose = observation_raw[:, 1:8]
        phase1_pose = observation_raw[:, 9:16]
        observation = np.hstack([slam_pose, phase1_pose])
        
        camera_path = pd.read_csv(camera_path).to_numpy()
        
        # For qos, we include the current slam observation & prediction

        res_image = []
        res_qpos = []

        for i in range(index, min(index + batch_size, len(observation))):
            qpos = np.asarray(observation[index])
            left_image = np.asarray(cv2.imread(camera_path[index][1]))
            right_image = np.asarray(cv2.imread(camera_path[index][3]))

            # Load the image from information
            all_cam_images = [left_image, right_image]
            all_cam_images = np.stack(all_cam_images, axis=0)

            image_data = torch.from_numpy(all_cam_images) / 255.0
            # qpos_data = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_data = torch.from_numpy(qpos).float().cuda()

            # image_data = torch.einsum('k h w c -> k c h w', image_data).cuda().unsqueeze(0)
            image_data = torch.einsum('k h w c -> k c h w', image_data).cuda()

            res_image.append(image_data)
            res_qpos.append(qpos_data)
        

        res_image = torch.stack(res_image)
        res_qpos = torch.stack(res_qpos)

        # return image_data, qpos_data
        return res_image, res_qpos
    
    def getGroundtruth(self):

        episode_id = len(MSD_LIST) - 1

        groundtruth_path = self.id2gtpath[episode_id]
        groundtruth_raw = pd.read_csv(groundtruth_path).to_numpy()
        groundtruth = groundtruth_raw[:, 1:]
        return groundtruth
    

    def getSlamSource(self):

        episode_id = len(MSD_LIST)-1

        observation_path = self.id2datasetpath[episode_id]
        observation_raw = pd.read_csv(observation_path).to_numpy()
        # Get only the poses  
        slam_pose = observation_raw[:, 1:8]

        return slam_pose


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_data_euroc(num_episodes, batch_size_train, batch_size_val):

    train_ratio = 0.9

    shuffled_indices = np.random.permutation(num_episodes)

    print("shuffled indices: ",shuffled_indices)

    # Prepare the data loader 
    episodeid2qposefile = {}
    episodeid2gtfile = {}
    episodeid2imagepath = {}

    for i in range(len(MSD_LIST)):
        episodeid2qposefile[i] = PROFILE_RESULT_MOTHER_FOLDER + MSD_LIST[i] + ALIGNED_POSE_SUFFIX
        episodeid2gtfile[i] = PROFILE_RESULT_MOTHER_FOLDER + MSD_LIST[i] + GROUNDTRUTH_SUFFIX
        episodeid2imagepath[i] = PROFILE_RESULT_MOTHER_FOLDER + MSD_LIST[i] + IMAGE_PATH_SUFFIX
    
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = EuroCStyleDataset(train_indices, episodeid2qposefile, episodeid2gtfile, episodeid2imagepath)
    val_dataset = EuroCStyleDataset(val_indices, episodeid2qposefile, episodeid2gtfile, episodeid2imagepath)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, None, train_dataset.is_sim

def load_test_euroc() -> EuroCStyleDataset:

    # Hardcoded as using the last one for doing the prediction
    i = len(MSD_LIST) - 1
    episodeid2qposefile = {}
    episodeid2gtfile = {}
    episodeid2imagepath = {}

    episodeid2qposefile[i] = PROFILE_RESULT_MOTHER_FOLDER + MSD_LIST[i] + ALIGNED_POSE_SUFFIX
    episodeid2gtfile[i] = PROFILE_RESULT_MOTHER_FOLDER + MSD_LIST[i] + GROUNDTRUTH_SUFFIX
    episodeid2imagepath[i] = PROFILE_RESULT_MOTHER_FOLDER + MSD_LIST[i] + IMAGE_PATH_SUFFIX
    
    val_dataset = EuroCStyleDataset([i], episodeid2qposefile, episodeid2gtfile, episodeid2imagepath)

    return val_dataset



    



### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
