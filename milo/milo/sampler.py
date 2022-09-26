from torch.multiprocessing import Pool
import copy
import math
import numpy as np
import time


def get_samples(env, policy, num_to_collect, seed, mode='samples', eval_mode=False, deepmimic=False):
    '''
    Function to collect samples in env using policy.

    Parameter env: gym env from which we collect samples
    Parameter policy: pytorch mjrl mlp policy
    Parameter num_to_collect: (int) The number of samples or trajectories to collect based on mode
    Parameter seed: (int) Base seed used to seed environment and numpy 
    Parameter mode: (string) Either "trajectories" or "samples". If "trajecotries", this function collects 
    <num_to_collect> trajectories, otherwise it collects at least <num_to_collect> samples. 
    Parameter eval_mode: (bool) When true, the mean output of the policy is used as the action. Otherwise,
                      the noisy output is used. 
    Parameter deepmimic: (bool) When true, we check the 'valid' flag in info that DeepMimic uses 
                       to determine if the trajectory is valid. DeepMimic considers a trajectory 
                       valid if the trajectory did not terminate due to the velocity being too large.
    
    '''

    paths_collected = 0
    samples_collected = 0
    seed_counter = 0

    paths = []
    if mode == 'trajectories':
        condition = lambda: paths_collected < num_to_collect
    else:
        condition = lambda: samples_collected < num_to_collect

    while condition():
        seed_counter += 1
        env.seed_env(seed + seed_counter)
        np.random.seed(seed + seed_counter)
        done = False
        valid = True
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        next_observations = []
        env_infos = []
        o = env.reset()
        ctr = 0
        while not done:
            a, agent_info = policy.get_action(o)
            a = agent_info['evaluation'] if eval_mode else a
            next_o, r, done, info = env.step(a)  # Take step
            observations.append(o)
            next_observations.append(next_o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(info)

            if deepmimic and not info['valid']:
                done = True
                valid = False

            o = next_o
            ctr += 1

        if valid:
            path = dict(
                observations=np.array(observations),
                next_observations=np.array(next_observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                agent_infos=stack_tensor_dict_list(agent_infos),
                env_infos=env_infos,
                # Don't stack because when env is gym_deepmimic, last element in env infos will contain dtw cost if imitate_amp.
                terminated=done
            )
            paths.append(path)
            paths_collected += 1
            samples_collected += ctr

    assert paths_collected == len(paths)
    return paths, samples_collected


def sample_points(env, policy, num_to_collect, base_seed, num_workers=4, mode='samples', eval_mode=False, verbose=False,
                  deepmimic=False):
    '''
    This function collects samples in env with policy using multiprocessing. 

    Parameter env: gym env from which we collect samples
    Parameter policy: pytorch mjrl mlp policy
    Parameter num_to_collect: (int) The number of samples or trajectories to collect based on mode
    Parameter base_seed: (int) Base seed used to seed environment and numpy 
    Parameter num_workers: (int) Numbers of workers to spawn using multiprocessing Pool. Each worker 
                        collects samples with the function get_samples             
    Parameter mode: (string) Either "trajectories" or "samples". If "trajecotries", this function collects 
    <num_to_collect> trajectories, otherwise it collects at least <num_to_collect> samples. 
    Parameter eval_mode: (bool) When true, the mean output of the policy is used as the action. Otherwise,
                      the noisy output is used. 
    Parameter verbose: (bool) If true, prints total number of samples and number of trajectories collected before returning. 
    Parameter deepmimic: (bool) When true, we check the 'valid' flag in info that DeepMimic uses 
                       to determine if the trajectory is valid. DeepMimic considers a trajectory 
                       valid if the trajectory did not terminate due to the velocity being too large.
    
    '''
    # TODO: Add code for if len(envs) ==1
    assert mode == 'samples' or mode == 'trajectories'

    
    #Create arguments for each process
    args = []
    num_per_thread = math.ceil(num_to_collect / num_workers)
    for i in range(num_workers):
        seed = 12345 + base_seed * i
        args.append((env, copy.deepcopy(policy), num_per_thread, seed, mode, eval_mode, deepmimic))
    
    #Spawm processes
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.starmap(get_samples, args)

    all_paths = []
    total_samples = 0
    for i in results:
        all_paths.extend(i[0])
        total_samples += i[1]
    if verbose:
        print(f"Collected {total_samples} and {len(all_paths)} trajectories in {time.time() - start_time} seconds")
    return all_paths


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    Parameter tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}

    Ex: tensor_dict_list = [{'hi':{'bye':1}}, {'hi':{'bye':2}}]
    returns: {'hi': {'bye': array([1, 2])}}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.array([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret
