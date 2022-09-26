import torch
import os
import random
import bisect
import numpy as np
from deepmimic.DeepMimic import update_timestep, build_world
from multiprocessing import Pool
from milo.logger import init_logger
import argparse
from tabulate import tabulate
from deepmimic.util.arg_parser import ArgParser
import sys
'''
This module contains functions used for collecting trajectories for the offline dataset inside the AMP framework. 
The models loaded should be the ones trained by AMP itself. This module is in the MILO directory instead of 
DeepMimic as this offline dataset will be used by MILO for training the dynamics model. 

The assumption is that the only scenes that will be loaded will be imitation-only scenes (either amp or deepmimic) with no goal
'''

def get_trajectory(world,  imitate_amp, record_stats, args, model_name):
    '''
    Sample one trajectory inside AMP framework.

    Parameter world: This should be an RLWorld object (see DeepMimic project).
    Parameter imitate_amp: (boolean) indicates whether the scene loaded is of type imitate_amp. The type of the scene
                        can be founded in the first line of the arg files used to load DeepMimic. Another way to check
                        this is check world.env.get_name().
    Parameter record_stats: (boolean) If true, we do not return the trajectory but instead return the scores/rewards and length of
                  of the trajectory
    Parameter args: command line arguments
    '''
    global update_timestep
    num_substeps = world.env.get_num_update_substeps()
    timestep = update_timestep / num_substeps

    time = np.random.uniform(args.time_min, args.time_max) if args.custom_time_range else np.random.uniform(0, world.env.get_motion_length())

    end_episode = False
    valid_episode = True
    world.reset_time(time, not args.no_resolve, args.noise_bef_rot, args.noise_min, args.noise_max, args.radian, args.rot_vel_w_pose, args.vel_noise, args.interpolation, args.knee_rot)

    # Update world following DeepMimic.update_world until episode ends. AMP framework handles collecting data.
    while not end_episode:
        world.update(timestep)
        valid_episode = world.env.check_valid_episode()
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if (end_episode):
                world.end_episode()
        else:
            end_episode = True

    if valid_episode:
        path = world.agents[0].path

        all_states = np.array(path.states)
        actions = np.array(path.actions)

        '''
        Uncomment this to get phi(s) for each state. This isn't a feature of the original DeepMimicCore and was
        added. Note the discriminator takes in as input (phi(s_t), phi(s_{t+1}) but it isn't the simple
        concatenation of the two vectors. Instead of first concatenates the pose part of phi(s_t) and phi(s_{t+1}) 
        and then the velocity parts so the input is a "zig-zagged" concatenation of the two phis. 

        amp_features = path.amp_obs_agent_current
        '''

        '''
        For AMP imitation only (no goal/task), the rewards returned by DeepMimicCore will be all 0's except the last 
        reward which actually corresponds to the DTW cost since there is no goal reward and the normal style reward is 
        not computed from DeepMimicCore but from the discriminator in learning/amp_agent.py. 

        Otherwise, if we are in DeepMimic imitation, the rewards vector contains r^I. 

        Note, we assume that there is no goal so we do not need to worry about recording the goal vector or
        reward containing r^G. This capability can be easily added. 
        '''
        # Return estimated DTW cost path if imitate amp, else return regular rewards.
        score_path = np.array(path.rewards) if not imitate_amp else np.array(world.env.get_dtw_backtrack_path(0))
        score = np.sum(score_path)

        #print(f"Diff is: {np.abs(score - np.sum(path.rewards))}, Reset time: {time}, score: {score}, rewards: {np.sum(path.rewards)}, model: {model_name} ")
        #assert np.abs(score - np.sum(path.rewards))<1, f"Reset time: {time}, score: {score}, rewards: {np.sum(path.rewards)}, model: {model_name}"
        if record_stats:
            return (score, len(actions))
        else:
            score_name = 'dtw_cost' if imitate_amp else 'ep_rew'
            return {'episode': (all_states, actions, score_path), score_name: score}
    else:
        return None


def get_data(args, files):
    '''
    Returns dictionary of statistics for models defined in files if recording stats. Otherwise, this function
    collects trajectories from the models in files and returns the trajectories as a list.

    Parameter args: Parsed command line arguments.
    Parameter files: (list) Each entry of this list is a tuple, x, where x[0] is the file of the model to load and x[1] is the
    number of trajectories to sample.

    '''
    world = build_world(['--arg_file', args.world_arg_file], enable_draw=False)
    scene_name = world.arg_parser.parse_string('scene')
    imitate_amp = scene_name == 'imitate_amp'
    output = {}
    record_stats = args.record_stats

    for (model, repetitions) in files:
        data = []
        print(f"Loading model: {model}")
        world.agents[0].load_model(model)
        count = 0
        while count < repetitions:
            trajectory = get_trajectory(world,  imitate_amp, record_stats, args, model)
            if trajectory is not None:
                data.append(trajectory)
                count += 1
        checkpoint_num = int(model[model.find('00000') + 5:model.find('ckpt') - 1])
        output[checkpoint_num] = data

    return output if record_stats else [traj for inner in output.values() for traj in inner]


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ######eneral parameters######
    parser.add_argument('--world_arg_file', type=str, help='path to Arg file used for initializing AMP imitation scene',
                        default='run_amp_humanoid3d_spinkick_args.txt')
    parser.add_argument('--num_procs', type=int, help='Number of processes. Each process will collect trajectories',
                        default=5)

    ######Collecting dataset arguments######
    parser.add_argument('--dataset_directory', type=str, default='../../data')
    parser.add_argument('--dataset_name', type=str, default='offline.pt')
    parser.add_argument('--stat_path', type=str,
                        help='path to stats that we will use to decide how many trajectories to sample from each model when creating dataset',
                        default='../../datasets/stats/stats.pt')

    ######Stat collection parameters######
    parser.add_argument('--record_stats',
                        help="When true, the stats of each model between lower_model and upper_model are recorded but we don't create the dataset",
                        action='store_true')
    parser.add_argument('--stat_repetitions', type=int,
                        help='Number of trajectories to sample for each model when collecting stats', default=50)
    parser.add_argument('--stat_save_directory', type=str, help='directory to store stats', default='../../data/stats')
    parser.add_argument('--stat_name', type=str, default='stats.pt')
    ######Model parameters######
    '''
    Currently, all models have the name agent0_int_model_00000xxxxx.ckpt where xxxxx is the iteration number (must be 5 digits)
    '''
    parser.add_argument('--lower_model', type=int, help='Iteration number of first model to test', default=0)
    parser.add_argument('--upper_model', type=int, help='Iteration number of last model to test', default=36200)
    parser.add_argument('--checkpoint_save_iters', type=int, help='This is how often the checkpoints were saved. Default is 200 so checkpoints were saved every 200 training steps.', default=200)
    parser.add_argument('--model_folder', type=str, help='Path to all the models',
                        default='../../deepmimic/deepmimic/output/intermediate/agent0_models')
    parser.add_argument('--model_prefix', type=str,
                        help='Prefix for models. Currently in AMP, models are saved to agent0_int_model_00000xxxxx.ckpt',
                        default='agent0_int_model_00000')

    '''
    Currently, the defaults are like this because the horizon of the spinkick is 10s or 300 timesteps. 
    This can be easily changed based on the horizon. 
    
    The way data is collected is that for each model, x,  to test in the range [args.lower_model, args.upper_model], 
    the number of trajectories we sample for for x is based on the stats we collected for that model in 
    args.stat_path. 
    
    Depending on what the average score (or length) of the trajectory for model x and where it falls in args.lower_limits,
    we index into args.repetitions to sample that many repetitions.
    
    For example, if model x has average score of 150, we find the index of the highest value in args.lower_limits
    that is less than 150. In this case, that would be index 1. We index into args.repetitions and sample 
    args.repetitions[1] samples for model x. 
    '''
    parser.add_argument('--lower_limits', nargs='+', help='Boundaries on limits on length for recording models',
                        type=int, default=[0, 100, 200])

    parser.add_argument('--repetitions', nargs='+',
                        help='Each entry tells us how many trajectories to sample for a model whose average limits falls within a specific boundary',
                        type=int, default=[300, 300, 300])
    parser.add_argument('--use_length_boundaries', action='store_true',
                        help='Currently, boundaries above are based on score (DTW for imitate_amp and reward otherwise. If this is true, we use length.')

    ######Trajectory reset parameters######
    parser.add_argument('--custom_time_range', action='store_true',
                        help='When true, resets to random time between time_min and time_max')
    parser.add_argument('--time_min', default=0, type=float, help='lower bound on reset time')
    parser.add_argument('--time_max', default=0, type=float, help='upper bound on reset time')
    parser.add_argument('--radian', type=float, default=0,
                        help='Angles in range [-radian, radian] radians are added to each joint. ')
    parser.add_argument('--noise_min', type=float, default=0, help='Lower bound on amount of noise added to state')
    parser.add_argument('--noise_max', type=float, default=0, help='Upper bound on amount of noise added to state')
    parser.add_argument('--no_resolve', action='store_true',
                        help='Resolve flag tells DeepMimicCore whether to resolve ground interesections when resolving. In most cases, should resolve')
    parser.add_argument('--noise_bef_rot', action='store_true',
                        help='There are two methods used to add randomness to state when resetting. This flag controls the order. Check DeepMimicCore KinCharacter.cpp for info on the two methods')
    parser.add_argument('--rot_vel_w_pose', action='store_true',
                        help='When true, any random roation applied during reset to pose is also applied to velocity')
    parser.add_argument('--vel_noise', action='store_true', help='When true, random noise added to velocity as well')
    parser.add_argument('--interpolation', type=float, default=1,
                        help='A float between [0,1] indicating how to initialize velocity during reset. 1 means to reset to expert. 0 is zero velocity')
    parser.add_argument('--knee_rot', action='store_true', help='When true, random noise added to knees in state')
    args = parser.parse_args()
    return args


def log_arguments(args, logger):
    """
    Adds arguments used for experiment in logger
    """
    headers = ['Args', 'Value']
    table = tabulate(list(vars(args).items()), headers=headers, tablefmt='pretty')
    logger.info(">>>>> Experiment Running with Arguments >>>>>")
    logger.info("\n" + table)


def main():
    args = get_args()

    # Check if scene type is imitate_amp
    arg_parser = ArgParser()
    arg_parser.load_file(args.world_arg_file)

    imitate_amp = arg_parser.parse_string('scene') == 'imitate_amp'

    #=====Create paths to all the model files and shuffle=====#
    prefix = os.path.join(args.model_folder, args.model_prefix)
    lower_checkpoint = args.lower_model
    upper_checkpoint = args.upper_model

    if args.record_stats:
        paths = [(prefix + f"{i:05}.ckpt", args.stat_repetitions) for i in
                 range(lower_checkpoint, upper_checkpoint + 1, 200)]
    else:
        stats = torch.load(args.stat_path)
        if args.use_length_boundaires:
            paths = [(prefix + f"{i:05}.ckpt",
                      args.repetitions[bisect.bisect_left(args.lower_limits, stats[i]['average_len']) - 1]) for i in
                     range(lower_checkpoint, upper_checkpoint + 1, 200)]
        else:
            if imitate_amp:
                paths = [(prefix + f"{i:05}.ckpt",
                          args.repetitions[bisect.bisect_left(args.lower_limits, stats[i]['average_dtw']) - 1]) for i in
                         range(lower_checkpoint, upper_checkpoint + 1, 200)]
            else:
                paths = [(prefix + f"{i:05}.ckpt",
                          args.repetitions[bisect.bisect_left(args.lower_limits, stats[i]['average_reward']) - 1]) for i
                         in range(lower_checkpoint, upper_checkpoint + 1, 200)]
    random.shuffle(paths)
    num_procs = min(args.num_procs, len(paths))



    if args.record_stats:
        #=====Collect statistics======#
        # Log args
        logger = init_logger(args.stat_save_directory, name='stats_' + args.dataset_name + '.log')
        log_arguments(args, logger)

        #=====Spawn processes using pool and rollout trajectories=====#
        results = []
        for i in range(0, len(paths), num_procs):
            x = min(len(paths) - i, num_procs)  # Number of processes to spawn in this iteration.
            arguments = [(args, [paths[i + j]]) for j in range(x)]
            with Pool(x) as pool:
                results.extend(pool.starmap(get_data, arguments))

        '''
        Results will be a list of dictionaries where in each dictionary, the keys are the model names and the values
        are a list of tuples (score,length) for that trajecctory. 
        '''
        output = {}
        for d in results:
            for k, v in d.items():
                # k = model name
                # v = list of tuples
                score = [x[0] for x in v]
                episode_lengths = [x[1] for x in v]
                if imitate_amp:
                    output[k] = {'average_dtw': np.average(score), 'std_dtw': np.std(score),
                                 'average_len': np.average(episode_lengths), 'std_len': np.std(episode_lengths)}
                else:
                    output[k] = {'average_rew': np.average(score), 'std_rew': np.std(score),
                                 'average_len': np.average(episode_lengths), 'std_len': np.std(episode_lengths)}

        save_directory = args.stat_save_directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        torch.save(output, os.path.join(save_directory, args.stat_name))


    else:
        #=====Collect dataset=====#
        logger = init_logger(args.dataset_directory, name=args.dataset_name + '.log')
        log_arguments(args, logger)
        results = []
        for i in range(0, len(paths), num_procs):
            x = min(len(paths) - i, num_procs)  # Number of processes to spawn in this iteration.

            x = num_procs if (i + num_procs <= len(paths)) else len(paths) - i
            arguments = [(args, [paths[i + j]]) for j in range(x)]
            with Pool(x) as pool:
                results.extend(pool.starmap(get_data, arguments))
        '''
        Results is a list of lists where each nested list contains dictionaries containing trajectories. 
        '''
        save_directory = args.dataset_directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        output = [j for inner in results for j in inner]
        torch.save(output, os.path.join(save_directory, args.dataset_name))

if __name__ == "__main__":
    main()
