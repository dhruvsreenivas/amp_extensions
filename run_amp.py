from deepmimic.env.deepmimic_env import DeepMimicEnv
from milo.utils import *
from milo.arguments import get_args
from milo.gail_cost import GAILCost
from milo.datasets import *

import numpy as np
import torch
import os
import gym
import time
import wandb
import gym_simenv
import gym_deepmimic
import json

from deepmimic.util.arg_parser import ArgParser
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.ppo_clip import PPO # in case we need to use this later

def main_amp():
    all_start = time.time()
    args = get_args()
    directories, _, _, loggers, writers, device = setup(args, amp=True, ask_prompt=False)
    
    loggers['amp'].info('Set up directories.')
    
    arg_parser = ArgParser()
    arg_parser.load_file(args.deepmimic)
    imitate_amp = arg_parser.parse_string('scene') == 'imitate_amp'
    env = DeepMimicEnv(['--arg_file', args.deepmimic], False)
    state_dim = env.get_state_size(0)
    action_dim = env.get_action_size(0)
    
    # real env to run things in
    reset_args = dict(custom_time=args.custom_time, time_min=args.time_min, time_max=args.time_max,
                      resolve=not args.no_resolve,
                      noise_bef_rot=args.noise_bef_rot, noise_min=args.noise_min, noise_max=args.noise_max,
                      radian=args.radian, rot_vel_w_pose=args.rot_vel_w_pose, vel_noise=args.vel_noise,
                      interp=args.interp, knee_rot=args.knee_rot)
    
    sim_train_env = gym.make('deepmimic-v0', deepmimic_args=args.deepmimic, train=True, reset_args=reset_args)
    sim_eval_env = gym.make('deepmimic-v0', deepmimic_args=args.deepmimic, train=False, reset_args=reset_args)
    
    loggers['amp'].info('Created training env.')
    
    # get expert data
    if args.cost_input_type == 'ss':
        expert_state, expert_next_state = get_db_mjrl(args.expert_data, expert=True, imitate_amp=imitate_amp)
    else:
        expert_state, expert_action, expert_next_state = get_db_mjrl(args.expert_data, imitate_amp=imitate_amp)
        
    loggers['amp'].info('Got expert data!')
    
    # log statistics of expert data
    n_expert_samples = expert_state.size(0)
    loggers['amp'].info(f'Number of expert samples: {n_expert_samples}')
    
    # set up replay buffer for storing agent data
    agent_rb = AgentReplayBuffer()
    
    # set up GAIL cost learnt online
    optim_args = {'lr': args.disc_lr, 'momentum': args.disc_momentum}
    if args.cost_input_type == 'ss':
        expert_data = torch.cat([expert_state, expert_next_state], dim=1)
        cost_fn = GAILCost(expert_data,
                           agent_rb=agent_rb,
                           feature_dim=1,
                           hidden_dims=[1024, 512],
                           input_type=args.cost_input_type,
                           scaling_coef=args.scaling_coef,
                           reg_coef=args.reg_coef,
                           lambda_b=args.lambda_b,
                           seed=args.seed,
                           disc_loss_type=args.disc_loss_type,
                           disc_opt=args.disc_opt,
                           disc_opt_args=optim_args)
    else:
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        cost_fn = GAILCost(expert_data,
                           agent_rb=agent_rb,
                           feature_dim=1,
                           hidden_dims=[1024, 512],
                           input_type=args.cost_input_type,
                           scaling_coef=args.scaling_coef,
                           reg_coef=args.reg_coef,
                           lambda_b=args.lambda_b,
                           seed=args.seed,
                           disc_loss_type=args.disc_loss_type,
                           disc_opt=args.disc_opt,
                           disc_opt_args=optim_args)
        
    loggers['amp'].info('Created AMP GAIL cost!')
    
    # agent components
    policy = MLP(state_dim,
                 action_dim,
                 hidden_sizes=tuple(args.actor_model_hidden),
                 seed=args.seed,
                 init_log_std=args.policy_init_log,
                 min_log_std=args.policy_min_log)
    
    baseline = MLPBaseline(state_dim,
                           reg_coef=args.vf_reg_coef,
                           batch_size=args.vf_batch_size,
                           hidden_sizes=tuple(args.critic_model_hidden),
                           epochs=args.vf_iters,
                           learn_rate=args.vf_lr)
     
    loggers['amp'].info('Training starting now...')
    start_training_time = time.time()
    print(f'Total time taken to set up: {start_training_time - all_start}')
    
    # let's do no bc to start, I don't think AMP does it
    if args.cost_input_type == 'ss':
        expert_paths = get_paths_mjrl(args.expert_data, expert=True)
    elif args.cost_input_type == 'sa':
        expert_paths = get_paths_mjrl(args.expert_data)
        
    # ========== agent training aside from BC warmstart ==========
        
    # agent init
    if not args.use_ppo:
        cg_args = {'iters': args.cg_iter, 'damping': args.cg_damping}
        planner_agent = NPG(sim_train_env,
                            policy,
                            baseline,
                            normalized_step_size=args.kl_dist,
                            hvp_sample_frac=args.hvp_sample_frac,
                            seed=args.seed,
                            FIM_invert_args=cg_args,
                            save_logs=True)
    else:
        planner_agent = PPO(sim_train_env,
                            policy,
                            baseline,
                            seed=args.seed,
                            save_logs=True)
    
    # training loop
    n_iter = 0
    best_policy_score = -float('inf')
    while n_iter < args.n_iter:
        # ======================== EVALUATING ========================
        loggers['amp'].info(f"\n {'=' * 20} Main Episode {n_iter + 1} {'=' * 20} \n")
        loggers['amp'].info("Evaluating AMP agent...")
        scores, costs = evaluate(n_iter, loggers['amp'], writers['amp'], args, sim_eval_env, planner_agent.policy,
                                 cost_fn, num_traj=args.num_eval_traj, imitate_amp=imitate_amp, gail_cost=args.gail_cost)

        save_and_plot(n_iter, args, directories, scores, costs)
        
        if scores['greedy'] > best_policy_score:
            best_policy_score = scores['greedy']
            save_checkpoint(directories, planner_agent, cost_fn, 'best', agent_type=args.planner)
        if (n_iter + 1) % args.save_iter == 0:
            save_checkpoint(directories, planner_agent, cost_fn, n_iter + 1, agent_type=args.planner)
        
        # ======================== TRAINING ========================
        # first train discriminator
        loggers['amp'].info(f"\n {'=' * 20} Discriminator updates happening now... {'=' * 20} \n")
        
        for _ in range(args.n_disc_update_steps):
            # sample data from replay buffer
            try:
                ss = agent_rb.sample(256)
            except Exception:
                print('Nothing in replay buffer: continuing to PG updates...')
                ss = None
                
            if ss is not None:
                print(f'We have {len(agent_rb)} examples in our replay buffer! \n')
                metrics = cost_fn.update_disc(ss)
                
                # log metrics
                loggers['amp'].info(f'Model GAIL total cost: {metrics["total_disc_loss"]}')
                loggers['amp'].info(f'Model GAIL expert disc loss: {metrics["expert_disc_loss"]}')
                loggers['amp'].info(f'Model GAIL MB disc loss: {metrics["model_based_disc_loss"]}')
                loggers['amp'].info(f'GAIL gradient penalty: {metrics["gradient_penalty"]}')
                loggers['amp'].info(f'GAIL weight regularizer loss: {metrics["regularizer_loss"]}')
                
                if 'expert_acc' in metrics.keys():
                    loggers['amp'].info(f'GAIL expert accuracy: {metrics["expert_acc"]}')
                    loggers['amp'].info(f'GAIL policy accuracy: {metrics["mb_acc"]}')
                
                # log chi^2 distance
                chi2_distance = cost_fn.compute_chi2_distance()
                loggers['amp'].info(f'Empirical chi^2 divergence over entire agent RB/expert dataset: {chi2_distance}')
        
        loggers['amp'].info(f'Discriminator update finished! \n')
        
        loggers['amp'].info(f"{'=' * 20} PG updates happening now... {'=' * 20} \n")
        best_baseline_optim, best_baseline = None, None
        curr_max_reward, curr_min_vloss = -float('inf'), float('inf')
    
        # now train policy
        for i in range(args.pg_iter):
            reward_kwargs = dict(gail_cost=args.gail_cost,
                                 reward_func=cost_fn,
                                 ensemble=None, # no model-based thing here
                                 device=device)
            
            planner_args = dict(N=args.samples_per_step, env=sim_train_env, sample_mode='model_based', \
                                gamma=args.gamma, gae_lambda=args.gae_lambda, num_cpu=args.num_cpu, \
                                eval_mode=args.eval_mode, verbose=False, agent_rb=agent_rb,
                                reward_kwargs=reward_kwargs) # TODO: FIX EVAL MODE
            
            _, _, _, _, _, infos = planner_agent.train_step(**planner_args)

            # Baseline Heuristic
            if infos['vf_loss_end'] < curr_min_vloss:
                curr_min_vloss = infos['vf_loss_end']
                best_baseline = planner_agent.baseline.model.state_dict()
                best_baseline_optim = planner_agent.baseline.optimizer.state_dict()
            
            # logging everything from PG iterations
            reward_mean = np.array(infos['reward']).mean()
            int_mean = np.array(infos['int']).mean()
            ext_mean = np.array(infos['ext']).mean()
            len_mean = np.array(infos['ep_len']).mean()
            loggers['amp'].info(f"alpha: {infos['alpha']}")
            loggers['amp'].info(f"kl_dist: {infos['kl_dist']}")
            loggers['amp'].info(f"npg_grad: {infos['npg_grad']}")
            loggers['amp'].info(f"vpg_grad: {infos['vpg_grad']}")
            loggers['amp'].info(f"Advantages: {infos['advantages']}")
            loggers['amp'].info(f"Old Dist Info: {infos['old_dist_info']}")
            loggers['amp'].info(f"New Dist Info: {infos['new_dist_info']}")
            loggers['amp'].info(f"Likelihood Ratio: {infos['lr']}")
            loggers['amp'].info(f"Surr before: {infos['surr_before']}")
            loggers['amp'].info(f"Surr after: {infos['surr_after']}")

            loggers['amp'].info(f"Running Score: {infos['running_score']}")
            loggers['amp'].info(f"Surr improvement: {infos['surr_improvement']}")

            ground_truth_mean = np.array(infos['ground_truth_reward']).mean()
            
            if not args.gail_cost:
                loggers['amp'].info(f'Model MMD: {infos["mb_mmd"]}')
                # loggers['amp'].info(f'Bonus MMD: {infos["bonus_mmd"]}')
                loggers['amp'].info(f'Model Ground Truth Reward: {ground_truth_mean}')
                loggers['amp'].info('PG Iteration {} reward | int | ext | ep_len ---- {:.2f} | {:.2f} | {:.2f} | {:.2f}' \
                                    .format(i + 1, reward_mean, int_mean, ext_mean, len_mean))
            
            save_sim_rewards((n_iter+1)*(i+1)*args.samples_per_step, directories, int_mean, ext_mean, len_mean)

            # Tensorboard Logging
            step_count = n_iter * args.pg_iter + i
            writers['amp'].add_scalar('data/reward_mean', reward_mean, step_count)
            writers['amp'].add_scalar('data/ext_reward_mean', ext_mean, step_count)
            writers['amp'].add_scalar('data/int_reward_mean', int_mean, step_count)
            writers['amp'].add_scalar('data/ep_len_mean', len_mean, step_count)
            writers['amp'].add_scalar('data/true_reward_mean', ground_truth_mean, step_count)
            writers['amp'].add_scalar('data/value_loss', infos['vf_loss_end'], step_count)
            writers['amp'].add_scalar('data/mb_mmd', infos['mb_mmd'], step_count)
            writers['amp'].add_scalar('data/bonus_mmd', infos['bonus_mmd'], step_count)
            
        planner_agent.baseline.model.load_state_dict(best_baseline)
        planner_agent.baseline.optimizer.load_state_dict(best_baseline_optim)
        n_iter += 1
        
    print(f"Total time to train: {time.time() - start_training_time}")
    loggers['milo'].info(f"Total time to train: {time.time() - start_training_time}")
    loggers['milo'].info(f"Total time overall: {time.time() - all_start}")
    
if __name__ == '__main__':
    main_amp()