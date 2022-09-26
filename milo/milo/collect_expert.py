import numpy as np
from deepmimic.env.deepmimic_env import DeepMimicEnv
import torch
import argparse
from deepmimic.util.arg_parser import ArgParser
import os
'''
This collects expert states from reference motions in the DeepMimic/AMP framework. There is no goal
information since that is unrelated to the reference motions. 
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--world_arg_file', type=str, help='Arg file used for initializing AMP imitation scene',
                        default='run_amp_humanoid3d_spinkick_args.txt')
    parser.add_argument('--no_resolve', action='store_true')
    parser.add_argument('--save_path', type=str, default='../../data')
    parser.add_argument('--num_traj', type=int, default=198)
    parser.add_argument('--save_name', type=str, default='expert.pt')
    args = parser.parse_args()

    #Load environment
    env = DeepMimicEnv(['--arg_file', args.world_arg_file], enable_draw=False)
    motion_length = env.get_motion_length()

    #Load arguments from scene argument file
    arg_parser = ArgParser()
    arg_parser.load_file(args.world_arg_file)
    horizon_seconds = arg_parser.parse_float('time_end_lim_max')
    assert horizon_seconds > 0, 'Please set the max time limit in the argument file'


    #Set timestep and number of timesteps
    time_per_step = 1/env.get_agent_update_rate()
    num_steps = int(horizon_seconds*env.get_agent_update_rate())
   
    '''
    The time_per_step should be 1/30. Additionally, this timestep is different than the timestep
    used in update_world. Here, we are not actually updating the bullet world but just stepping forward
    to get the expert state. In DeepMimic with the bullet world, the world was updated at 1.2kHz. Here, 
    the policy is queried at 30Hz so we just want to call update_sync_timestep with 1/30 since that
    gets the expert after timestep 1/30. 
    '''

    output = []
    #####Reset at index 0 and rollout the "even" trajectory#####
    states = []
    env.reset_index(0, not args.no_resolve);
    states.append(env.record_state(0))
    for i in range(num_steps):
        env.update_sync_timestep(time_per_step)
        states.append(env.record_state(0))


    output.append({'episode':states})

    #####Reset at index 1 and rollout the "odd" trajectory#####
    states= []
    env.reset_index(1, not args.no_resolve);
    states.append(env.record_state(0))
    for i in range(num_steps):
        env.update_sync_timestep(time_per_step)
        states.append(env.record_state(0))
    output.append({'episode':states})

    #####Rollout by resetting to random time#####
    for _ in range(args.num_traj):
        states = []
        start_time = np.random.uniform(low=0.0, high=motion_length)
        env.reset_time(start_time, not args.no_resolve)
        for i in range(num_steps):
            env.update_sync_timestep(time_per_step)
            states.append(env.record_state(0))
        output.append({'episode':states})
    
    if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    torch.save(output, os.path.join(args.save_path, args.save_name))

