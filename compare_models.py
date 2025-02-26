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

import pandas as pd
# from dcor import distance_correlation
from sklearn.cross_decomposition import CCA






import sys
sys.path.append('/home/wuhaolu/Documents/pose_prediction/')
from PosePrediction.utils import *
from PosePrediction.predictPoseImpl import get_trimmed_input_from_phase1, get_trimmed_input_from_slam, prepareTestSets


import joblib

import csv
import time
import pandas as pd

import sys
sys.path.append('/home/wuhaolu/Documents/pose_prediction/PosePrediction')
from utils import * 



prefix_list = [
    "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/ACT_Result/full_euler_angle/",
    # "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/ACT_Result/simple_transformer_tiny/"
]

traditional_model_list = [
    "/home/wuhaolu/Documents/pose_prediction/PosePrediction/DTRModle.joblib",
    "/home/wuhaolu/Documents/pose_prediction/PosePrediction/LinearModel.joblib"
]

# We need to have the following criteria:
# 1. the 3-axis angle diviation in the previous 100ms
# 2. the eludician variance in the previous 100ms
# 3. so we need to profile the groundtruth to get the 

def compute_delta_gap(gt, window_size):

    # Then we compute the window for the
    # 
    prev_gt = gt[:len(gt) - window_size]
    cur_gt = gt[window_size:]

    pose_diff, angle_diff = computePoseDiffFromNumpy(prev_gt, cur_gt)

    return pose_diff, angle_diff


def combine_indices_to_intervals(indices):
    if not indices:
        return []

    # Sort the indices
    indices = sorted(indices)

    intervals = []
    start = indices[0]
    end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:  # Consecutive number
            end = indices[i]
        else:  # Gap detected, finalize the current interval
            if end - start >= 10: 
                intervals.append((start, end))
            start = indices[i]
            end = indices[i]

    # Add the last interval
    intervals.append((start, end))
    return intervals

# See whether there are certain behavior that under certain scenairos, linear is better or the same as advanced models
def compare_models(model1_pose_error, model2_pose_error, model1_angle_error, model2_angle_error):

    min_len = min(len(model1_pose_error), len(model2_pose_error))
    model1_pose_error = model1_pose_error[:min_len]
    model2_pose_error = model2_pose_error[:min_len]
    
    model1_better_pose_index = []
    model1_better_yaw_index = []
    model1_better_pitch_index = []
    model1_better_roll_index = []

    # Then we compare the results with the same as 
    for i, (model1_pose, model2_pose) in enumerate(zip(model1_pose_error, model2_pose_error)):

        if model1_pose < model2_pose:
            model1_better_pose_index.append(i)
    
    for i, (model1_angle, model2_angle) in enumerate(zip(model1_angle_error, model2_angle_error)):

        if model1_angle[0] < model2_angle[0]:

            model1_better_yaw_index.append(i)
        
        if model1_angle[1] < model2_angle[1]:

            model1_better_pitch_index.append(i)
        
        if model1_angle[2] < model2_angle[2]:
            model1_better_roll_index.append(i)
    
    model1_better_pose_index = combine_indices_to_intervals(model1_better_pose_index)
    model1_better_yaw_index = combine_indices_to_intervals(model1_better_yaw_index)
    model1_better_pitch_index = combine_indices_to_intervals(model1_better_pitch_index)
    model1_better_roll_index = combine_indices_to_intervals(model1_better_roll_index)
    
    return model1_better_pose_index, model1_better_yaw_index, model1_better_pitch_index, model1_better_roll_index


def fetch_result(args):

    policy_class = args['policy_class']

    # Read the groundtruth
    dataset = load_test_euroc(TEST_IDX, policy_class)
    groundtruth = dataset.getGroundtruth()
    slam_output = dataset.getSlamSource()

    # Fetch the ACT results

    act_diff_list = []
    traditional_diff_list = []

    for window in prediction_window:

        # Get the transformer models
        for prefix in prefix_list:
            actions = pd.read_csv(prefix + "res_" + str(window) + ".csv").to_numpy()[:,:6]
            print(len(actions))

            groundtruth = groundtruth[window: min(len(actions) + window, len(groundtruth))]
            actions = actions[:len(groundtruth)]

            print(actions.shape, groundtruth.shape)
            act_diff_list.append(computePoseDiffFromNumpy(actions, groundtruth))

    # Fetch traditional model results

    raw_test_sets_pair = prepareTestSets()
    
    # Currently, we only interested in the gap of 100ms
    idx_list = [2, 4]
    test_sets_pair = []
    for idx in idx_list:
        test_sets_pair.append(raw_test_sets_pair[idx])
    
    # Then we do model prediction on those results
    for model_path in traditional_model_list:

        model = joblib.load(model_path)

        for _, test_x, _, test_y in test_sets_pair:

            print(test_x[0])
            # Generate the prediction results of error ness
            traditional_diff_list.append(model.predict(test_x, test_y))
    

    '''We got the result for traditional and act results, then we can compare and see the difference'''

    for i, (act_pose_diff, act_angle_diff) in enumerate(act_diff_list):

        for j, (traditional_pose_diff, traditional_angle_diff) in enumerate(traditional_diff_list):

            res = compare_models(traditional_pose_diff, act_pose_diff, traditional_angle_diff, act_angle_diff)
            print(res[0])
            print(res[1])
            print(res[2])
            print(res[3])
            print("-------------------")



def dumpIMUInfo():
    
    source_infos = None
    target_infos = None
    
    for file in [MSD_LIST[TEST_IDX]]:
            
        source_info = fetchIMUInfoFromFile(file)
        target_info = fetchSlamErrorFromFile(file)

        if source_infos is None:  
            source_infos = source_info
            target_infos = target_info 
        else:
            source_infos = np.vstack([source_infos, source_info])
            target_infos = np.vstack([target_infos, target_info])

        # source_infos.append(source_info)
        # target_infos.append(target_info)

    source_infos = np.asarray(source_infos)
    target_infos = np.asarray(target_infos)

    print(source_infos.shape)
    print(target_infos.shape)

    with open("sourceinfo.csv", "w") as source_out:
        
        for i in range(source_infos.shape[1]-1):
            source_out.write(str(i) +", ")
        source_out.write(str(i) + "\n")

        csv_writer = csv.writer(source_out)
        csv_writer.writerows(source_infos)
    
    with open("targetinfo.csv", "w") as target_out:
        
        for i in range(target_infos.shape[1]-1):
            target_out.write(str(i) +", ")
        target_out.write(str(i) +"\n")

        csv_writer = csv.writer(target_out)
        csv_writer.writerows(target_infos)

from scipy.stats import pearsonr, spearmanr


def drawSlamPearsonCorrelation():
    source_infos = pd.read_csv("sourceinfo.csv").to_numpy()
    target_infos = pd.read_csv("targetinfo.csv").to_numpy()
    print(source_infos.shape)
    print(target_infos.shape)

    data = pd.DataFrame(np.hstack([source_infos, target_infos]), 
                        columns=[f"x{i+1}" for i in range(source_infos.shape[1])] + [f"y{j+1}" for j in range(target_infos.shape[1])])
    
    # source_len = source_infos.shape[1]
    source_len = 7
    target_len = target_infos.shape[1]

    source_name = ["time", 
                   "X", "Y", "Z", 
                   "yaw", "pitch", "roll", 
                   "X_variance", "Y_variance", "Z_variance", 
                   "yaw_variance", "pitch_variance", "roll_variance"]
    target_name = ["translation", "yaw", "pitch", "roll"]

    fig, axs = plt.subplots(source_len, target_len, figsize=(20, 20))

    for i in range(source_len):
        for j in range(target_len):

            ax = axs[i, j]
            ax.scatter(data.iloc[:, i], data.iloc[:, j + source_len])
            # ax.set_title(f"X{i+1} vs Y{j+1}")
            ax.set_xlabel(source_name[i])
            ax.set_ylabel(target_name[j])
    
    fig.savefig("PearsonCorrelation.png")
    


def testSlamAccuracyRelationshipWithIMU():

    source_infos = pd.read_csv("sourceinfo.csv").to_numpy()
    target_infos = pd.read_csv("targetinfo.csv").to_numpy()
    print(source_infos.shape)
    print(target_infos.shape)

    time = source_infos[:, 0]
    dist = target_infos[:, 0]

    time_dist_pair = sorted(zip(time, dist), key=lambda x: x[0])

    time = [x[0] for x in time_dist_pair]
    dist = [x[1] for x in time_dist_pair]

    # plt.plot(time, dist)
    # plt.show()

    data = pd.DataFrame(np.hstack([source_infos, target_infos]), 
                        columns=[f"x{i+1}" for i in range(source_infos.shape[1])] + [f"y{j+1}" for j in range(target_infos.shape[1])])
    
    
    pearson_corr_matrix = data.corr(method='pearson')
    print("Pearson Correlation Matrix:")
    print(pearson_corr_matrix.iloc[source_infos.shape[1]:, :source_infos.shape[1]])

    # Compute Spearman correlation matrix
    spearman_corr_matrix = data.corr(method='spearman')
    print("Spearman Correlation Matrix:")
    print(spearman_corr_matrix.iloc[source_infos.shape[1]:, :source_infos.shape[1]])


    # Distance Correlationy
    
    target_size = target_infos.shape[1]
    target_name = ["translation error", "yaw error", "pitch error", "roll error"]

    varial_name = ["linear_acceleration", "angular_acceleration", "linear_variance", "angular_variance"]
    
    
    fig, axs = plt.subplots(target_size, 4, figsize=(20, 20))

    fig_time, axs_time = plt.subplots(target_size, 4, figsize=(30, 20))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Each i represents an error type (Target)
    for i in range(target_infos.shape[1]):
        print(i)

        cca = CCA(n_components=1)
        Y = target_infos[:, i:i+1]
        # print(source_infos[:,1:4].shape, Y.shape)
        X_scores, Y_scores = cca.fit_transform(source_infos[:,0:1], Y)

        corrs = [np.corrcoef(X_scores[:, i], Y_scores[:, i])[0, 1] for i in range(1)]    
        print(corrs)

        for k, j in enumerate([1, 4, 7, 10]):
            
            cca = CCA(n_components=1)
            Y = target_infos[:, i:i+1]
            # print(source_infos[:,1:4].shape, Y.shape)
            # [1:3] linear accel average
            # [3:6] anglar accel average
            # [6:9] linear variance
            # [9:12] angular variance
            X_scores, Y_scores = cca.fit_transform(source_infos[:,j:j+3], Y)

            ax = axs[i, k]

            # plt.figure(figsize=(8, 6))
            ax.scatter(X_scores, Y_scores, alpha=0.4, edgecolors='k')
            # ax.set_title('Scatter Plot of Canonical Variables', fontsize=14)
            ax.set_xlabel(varial_name[k], fontsize=12)
            ax.set_ylabel(target_name[i], fontsize=12)

            ax_time = axs_time[i, k]

            time_series = np.arange(len(X_scores))

            ax_time.scatter(time_series, X_scores, alpha=0.4)
            ax_time.set_xlabel("Time", fontsize=12)
            ax_time.set_ylabel(varial_name[k], fontsize=12)
            

            ax_time2 = ax_time.twinx()
            ax_time2.scatter(time_series, Y_scores, alpha=0.4, color='red')
            ax_time2.set_ylabel(target_name[i], fontsize=12)

            # # Print the score
            # print(score)
            corrs = [np.corrcoef(X_scores[:, i], Y_scores[:, i])[0, 1] for i in range(1)]    
            print(corrs)
        print("----------")
    fig.savefig("CCACorrelation.png")
    fig_time.savefig("CCACorrelationTime.png")



def compute_result_from_euler_file(args):

    policy_class = args['policy_class']

    # Read the groundtruth
    dataset = load_test_euroc(TEST_IDX, policy_class)
    groundtruth = dataset.getGroundtruth()
    slam_output = dataset.getSlamSource()

    pose_diff_computed_slam_and_gt_slam, \
        angle_diff_computed_slam_and_gt_slam = computePoseDiffFromNumpy(slam_output, groundtruth)

    for window in prediction_window:
        pose_diff_list = []
        yaw_list = []
        pitch_list = []
        roll_list = []

        # Get the transformer models
        for prefix in prefix_list:
            
            # Res contains the prediction result for the given window size
            # So we can claim that the result is: the result of predicting i+window at i
            # And so the result is related with:
            # [i-window, i]
            # The slam source error at i, and for the starting [:window], the previous error is not available
            actions = pd.read_csv(prefix + "res_" + str(window) + ".csv").to_numpy()[:,:6]
            print(len(actions))
            
            # Get the groundtruth from the previous window size
            history_len = window // 2
            gt_pose_diff_from_previous_window, angle_diff_from_previous_window = compute_delta_gap(groundtruth, history_len)
            
            # Get the slam source error rate for each window
            # This is the slam source error for pose at each groundtruth level

            # Get the groundtruth for source error
            groundtruth = groundtruth[window: min(len(actions) + window, len(groundtruth))]
            actions = actions[:len(groundtruth)]

            print(actions.shape, groundtruth.shape)
            pose_diff, angle_diff = computePoseDiffFromNumpy(actions, groundtruth)
            
            # Valid len contains result from [windows_size: len - windows_size]
            
            valid_len =  len(gt_pose_diff_from_previous_window)
            pose_diff_len = min(history_len + valid_len, len(pose_diff))
            pose_diff = pose_diff[history_len: pose_diff_len]
            angle_diff = angle_diff[history_len: pose_diff_len]
            
            pose_diff_computed_slam_and_gt_slam = pose_diff_computed_slam_and_gt_slam[history_len: len(pose_diff) + history_len]
            
            gt_pose_diff_from_previous_window = gt_pose_diff_from_previous_window[:len(pose_diff)]
            angle_diff_from_previous_window = angle_diff_from_previous_window[: len(pose_diff)]

            print(gt_pose_diff_from_previous_window.shape)
            print(pose_diff.shape)
            print(angle_diff.shape)
            print(pose_diff_computed_slam_and_gt_slam.shape)

            # So we have those params candidate:
            # 1. xyz error rate corresponding to the final error rate

            yaw = angle_diff[:, 0]
            pitch = angle_diff[:, 1]
            roll = angle_diff[:, 2]

            # interval = len(pose_diff)
            interval = 1000
            for i in range(0, len(pose_diff), interval):

                change_from_previous_window = gt_pose_diff_from_previous_window[i: i+interval]
                yaw_change_from_previous_window = angle_diff_from_previous_window[i: i+interval, 0]
                pitch_change_from_previous_window = angle_diff_from_previous_window[i: i+interval, 1]
                roll_change_from_previous_window = angle_diff_from_previous_window[i: i+interval, 2]

                pose_diff_cur_interval = pose_diff[i:i+interval]
                yaw_cur_interval = yaw[i:i+interval]
                pitch_cur_interval = pitch[i:i+interval]
                roll_cur_interval = roll[i:i+interval]

                # Plot based on different combination

                observations = [change_from_previous_window, 
                                yaw_change_from_previous_window, 
                                pitch_change_from_previous_window, 
                                roll_change_from_previous_window]
                
                errors = [pose_diff_cur_interval, 
                          yaw_cur_interval, 
                          pitch_cur_interval, 
                          roll_cur_interval]
                
                fig, axs = plt.subplots(4, 4)

                for k, observation in enumerate(observations):

                    for j, error in enumerate(errors):

                        # Plot and see whether there are certain errors
                        # Sort according to observations
                        
                        observation_error_pair = list(zip(observation, error))
                        sort_res = sorted(observation_error_pair, key= lambda x: x[0])
                        # print(sort_res)

                        sorted_observation = [x[0] for x in sort_res]
                        sorted_error = [x[1] for x in sort_res]
                        axs[k][j].plot(sorted_observation, sorted_error)
                fig.savefig(prefix + "error_distribution_" + str(i) + ".png")


            # The key thing is that: is there any chance that certain prediction is better than the others under certain scenarios
            # And if so, what will be the 

            pose_diff_list.append(pose_diff)
            yaw_list.append(yaw)
            pitch_list.append(pitch)
            roll_list.append(roll)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)

    # compute_result_from_euler_file(vars(parser.parse_args()))
    # fetch_result(vars(parser.parse_args()))
    dumpIMUInfo()
    drawSlamPearsonCorrelation()
    testSlamAccuracyRelationshipWithIMU()
        