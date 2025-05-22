import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
# import wandb
import time
from torchvision import transforms
import signal
import sys

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE
import h5py
from pathlib import Path

import IPython
e = IPython.embed

from test_robot import Xarm
from video_receiver import VideoStreamer

# import xarm sdk

import os
import sys
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../xArm-Python-SDK'))

from xarm.wrapper import XArmAPI



def get_test_data(i = 0):
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir / '../human2robot/grab_cube2_v2/episode_test.hdf5'
    with h5py.File(file_path,'r') as f:
        qpos = f['qpos'][i]
        image = f['cam_data']['robot_camera'][i]
        
    return qpos, image
        


def run(config, ckpt_name, save_episode=True, num_rollouts=50):
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    # max_timesteps = 200   #TODO: hardcode 
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None




    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    
    test = False
    # TODO you should set the robot here
    robot = Xarm("10.177.63.209")
    print(robot.get_cartesian_position())
    def handler(signum, frame):
        robot.move_gripper_percentage(1)
        robot.home()
        
        robot.stop()
        exit(1)

    signal.signal(signal.SIGINT, handler)
    
    try:
        video_streamer = VideoStreamer("10.177.63.229", 10005)
        result = video_streamer.get_image_tensor()
        print(result.shape)
    except:
        print("camera not on")
    
    # try:
    #     from configparser import ConfigParser
    #     parser_robot = ConfigParser
    #     robot_conf_path = Path(__file__).resolve().parent / './robot.conf'
    #     print(robot_conf_path)
    #     parser_robot.read(robot_conf_path)
    #     ip = parser_robot.get('xArm', 'ip')
    # except:
    #     ip = input('Please input the xArm ip address:')
    #     if not ip:
    #         print('input error, run this file in test mode')
    #         # sys.exit(1)  
    #         test = True
    #     else:
    #         arm = XArmAPI(ip, is_radian = False)
    #         arm.motion_enable(enable=True)
    #         arm.set_mode(0)
    #         arm.set_state(state=0)
            
    
    
    env_max_reward = 0
    
    
    query_frequency = policy_config['num_queries']
    
    BASE_DELAY = 3
    # 获取frequency的频率
    query_frequency -= BASE_DELAY
    
    max_timesteps = int(max_timesteps * 1)
    
    episode_returns = []
    highest_reward = []
    
    
    # Just for TEST
    # TODO you should remove it after test
    if test:
        num_rollouts = 1
    
    for rollout_id in range(num_rollouts):
        
        
        
        # TODO you should reset your robot here
        robot.move_gripper_percentage(1)
        robot.home()
        
        e()
        
        if onscreen_render:
            # TODO you can render the camera data on screen here
            pass
        
        image_list = []
        qpos_list = []
        
        print(f'max_timesteps: {max_timesteps}')
        
        # evaluation loop
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                time1 = time.time()
                
                if onscreen_render:
                    # TODO you can render the camera data on screen here
                    pass
                
                time2 = time.time()
                
                ### TODO you should get the iamge and qpos here form robot
                if test == False:
                    image = video_streamer.get_image_tensor()
                    # qpos = robot.get_cartesian_position()
                    # qpos = robot._controller.last_used_angles
                    qpos = robot.get_cartesian_position()
                    gripper_state = robot.get_gripper_state()
                    qpos = np.append(qpos, gripper_state)
                    # image = ? what is the api
                else:
                    qpos, image = get_test_data(t)
                
                # data processing
                image_list.append(image)
                qpos_list.append(qpos)
                
                qpos_numpy = np.array(qpos)
                qpos = pre_process(qpos)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                image = rearrange(image, 'h w c -> c h w')
                image = torch.from_numpy(image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)
                
                print(f'image: {image.shape}, pos: {qpos.shape}')
                
                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos=qpos, image=image)
                    print('network warm up done')
                    time1 = time.time()
                
                
                ### query policy
                time3 = time.time()
                if t % query_frequency == 0:
                    all_actions = policy(qpos=qpos, image=image)

                    # print(f'{all_actions.shape=}')
                    # input('wait')
                    all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    print(f'all actions: {all_actions.shape}')
                
                time4 = time.time()   
                raw_action = all_actions[:, t % query_frequency]
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-4]
                target_gripper_state = action[-3]
                base_action = action[-2:]
                
                
                
                ### step the robot
                time5 = time.time()
                # TODO you should use the qpos and gripper state
                # to control your robot here
                print(f'target_qpos: {target_qpos}, target_gripper_state: {target_gripper_state}')
                
                if(target_gripper_state >= 0.5):
                    target_gripper_state = 1 # 张开
                else:
                    target_gripper_state = 0
                # target_gripper_state = max(0, min(1, target_gripper_state))
                # robot._controller.set_servo_angle(angle=target_qpos, speed = 50,is_radian=False, wait=True)
                # target_qpos = np.mod(target_qpos, 360)
                target_qpos = target_qpos.astype(int)
                
                
                if test == False:
                    robot.move_coords(target_qpos)
                    robot.move_gripper_percentage(target_gripper_state)
                    # robot.move(target_qpos)
                    
                
                # if test == False:
                #     code = arm.set_servo_angle(angle=target_qpos, speed = 50,is_radian=False, wait=True)
                #     # gripper = ? what is the api
                
                
        # TODO you can log the data of this rollout here,
        # and virualize it if you want
        print(f'Rollout {rollout_id} finished')
        
    robot.stop()
        
    # TODO implement return value here
    return 0, 0
                
                
def main(args):
    ckpt_dir = args['ckpt_dir']
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        
    # ckpt_name = 'policy_best.ckpt'
    ckpt_name = 'policy_step_1000_seed_0.ckpt'

    success_rate, avg_return = run(config, ckpt_name, save_episode=True, num_rollouts=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    
    main(vars(parser.parse_args()))
            
                