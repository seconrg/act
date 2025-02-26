import torch
from imitate_episodes import make_policy
from act.utils import *
import argparse

sourceCkptPath = '/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/ACT_Result/full_quat_100/'
packet_name = 'policy_best.ckpt'
policy_name = "ACT"
policy_config_act = {'lr': 1e-5,
                     'num_queries': 100,
                     'kl_weight': 10,
                     'hidden_dim': 512,
                     'dim_feedforward': 3200,
                     'lr_backbone': 1e-5,
                     'backbone': 'resnet18',
                     'enc_layers': 4,
                     'dec_layers': 7,
                     'nheads': 8,
                     'camera_names': ['left', 'right'],
                    }

# Initialize your model architecture
from policy import ACTPolicy, CNNMLPPolicy, SimplePolicy

def prepare_config():
    set_seed(1)
    # command line parameters
    is_eval = True
    ckpt_dir = sourceCkptPath
    policy_class = policy_name
    onscreen_render = False
    task_name = 'sim_vr_pose'
    batch_size_train = 8
    batch_size_val = 8
    num_epochs = 4000

    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    state_dim = DIM
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': 1e-5,
                         'num_queries': 100,
                         'kl_weight': 10,
                         'hidden_dim': 512,
                         'dim_feedforward': 3200,
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
        'lr': 1e-5,
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': 0,
        'temporal_agg': False,
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    return config
    


def dumpPolicy():

    config = prepare_config()

    policy_class = config['policy_class']
    policy_config = config['policy_config']

    ckpt_path = sourceCkptPath + packet_name
    policy = make_policy(policy_class, policy_config=policy_config)

    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    policy.eval()

    scripted_model = torch.jit.script(policy)
    print("scripted: ", scripted_model)
    # torch.jit.save(scripted_model, "model.pt")
    scripted_model.save("model.pt")
    



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    # parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    # parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    # parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # # for ACT
    # parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # parser.add_argument('--temporal_agg', action='store_true')
    
    dumpPolicy()