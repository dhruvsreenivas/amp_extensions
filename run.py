from deepmimic.env.deepmimic_env import DeepMimicEnv
from milo.utils import *
from milo.arguments import get_args
from milo.dynamics import DynamicsEnsemble
from milo.datasets import AmpDataset
from milo.linear_cost import MLPCost, RBFLinearCost

import numpy as np
import torch
import os
import gym
import gym_simenv
import gym_deepmimic
import json

from deepmimic.util.arg_parser import ArgParser
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.behavior_cloning import BC
from mjrl.algos.npg_cg import NPG
import time

def main():
    all_start = time.time()
    args = get_args()
    milo_directories, dynamic_directories, load_dynamics, loggers, writers, device = setup(args, ask_prompt=False)
    # = torch.device('cpu')  # https://pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations currently need to force cpu on windows.
    # Create deepmimic core env

    arg_parser = ArgParser()
    arg_parser.load_file(args.deepmimic)
    imitate_amp = arg_parser.parse_string('scene') == 'imitate_amp'
    env = DeepMimicEnv(['--arg_file', args.deepmimic], False) #Comment out and hardcode in state/action size so we don't need to load in deepmimic

    state_size = env.get_state_size(0)
    action_size = env.get_action_size(0)

    # Load Datasets
    offline_state, offline_action, offline_next_state = get_db_mjrl(args.offline_data, imitate_amp=imitate_amp)
    offline_dataset = AmpDataset(offline_state, offline_action, offline_next_state, device)

    # Validate dataset for training dynamics model
    if args.validate:
        validate_state, validate_action, validate_next_state = get_db_mjrl(args.validate_data, imitate_amp=imitate_amp)
        validate_dataset = AmpDataset(validate_state, validate_action, validate_next_state, device)
    else:
        validate_dataset = None


    # ==========Create Dynamic Model=========== #
    if args.dynamic_optim == 'sgd':
        optim_args = {'optim': 'sgd', 'lr': args.dynamic_lr, 'momentum': args.dynamic_momentum}
    elif args.dynamic_optim == 'adam':
        optim_args = {'optim': 'adam', 'lr': args.dynamic_lr, 'eps': args.dynamic_eps}
    else:
        assert False, "Choose a valid optimizer"


    # ==============Train dynamic model============== #
    if load_dynamics:
        loggers['dynamic'].info(f'>>>>Loading Dynamics model')
        ensemble_path = args.dynamic_checkpoint if args.dynamic_checkpoint else os.path.join(
            dynamic_directories['models_dir'], 'ensemble.pt')
        #Force load to cpu
        device = torch.device('cpu')
        loggers['dynamic'].info(f">>>>Device moved to {device}")
        dynamic_ensemble = DynamicsEnsemble(state_size, action_size, offline_dataset, validate_dataset,
                                            num_models=args.dynamic_num_models,
                                            batch_size=args.dynamic_batch_size, hidden_sizes=args.dynamic_hidden_sizes,
                                            transform=args.dynamic_transform,
                                            dense_connect=args.dynamic_dense_connect, optim_args=optim_args,
                                            base_seed=args.seed, device=device)
        dynamic_ensemble.load_ensemble(ensemble_path)
    else:
        loggers['dynamic'].info(f">>>>> Training Dynamics model on {device}")
        dynamic_ensemble = DynamicsEnsemble(state_size, action_size, offline_dataset, validate_dataset,
                                            num_models=args.dynamic_num_models,
                                            batch_size=args.dynamic_batch_size, hidden_sizes=args.dynamic_hidden_sizes,
                                            transform=args.dynamic_transform,
                                            dense_connect=args.dynamic_dense_connect, optim_args=optim_args,
                                            base_seed=args.seed, device=device)

        dynamic_ensemble.train(args.dynamic_epochs, validate=args.validate, logger=loggers['dynamic'], log_epoch=True,
                               grad_clip=args.dynamic_grad_clip,
                               save_path=dynamic_directories['models_dir'] if args.dynamic_save_models else None,
                               save_checkpoints=args.dynamic_save_checkpoints,
                               writer=writers['dynamic'])
        dynamic_ensemble.save_ensemble(os.path.join(dynamic_directories['models_dir'], 'ensemble.pt'))

        # Reload dynamics model if model was trained on gpu. Force device to be cpu since milo IL will be on cpu.
        if device != torch.device('cpu'):
            device = torch.device('cpu')
            loggers['dynamic'].info(f">>>>Device moved to {device}")
            dynamic_ensemble = DynamicsEnsemble(state_size, action_size, offline_dataset, validate_dataset,
                                                num_models=args.dynamic_num_models,
                                                batch_size=args.dynamic_batch_size, hidden_sizes=args.dynamic_hidden_sizes,
                                                transform=args.dynamic_transform,
                                                dense_connect=args.dynamic_dense_connect, optim_args=optim_args,
                                                base_seed=args.seed, device=torch.device('cpu'))
            dynamic_ensemble.load_ensemble(os.path.join(dynamic_directories['models_dir'], 'ensemble.pt'))

    # Get threshold
    dynamic_ensemble.compute_threshold()

    # ==============Create Envs==============#

    # Load params needed for simenv initialization
    reset_args = dict(custom_time=args.custom_time, time_min=args.time_min, time_max=args.time_max,
                      resolve=not args.no_resolve,
                      noise_bef_rot=args.noise_bef_rot, noise_min=args.noise_min, noise_max=args.noise_max,
                      radian=args.radian, rot_vel_w_pose=args.rot_vel_w_pose, vel_noise=args.vel_noise,
                      interp=args.interp, knee_rot=args.knee_rot)

    mb_env = gym.make('simenv-v0', deepmimic_args=args.deepmimic, dynamic_ensemble=dynamic_ensemble, reset_args=reset_args)
    inf_env = gym.make('deepmimic-v0', deepmimic_args=args.deepmimic, reset_args=reset_args)

    # ==============Create Costs==============#
    if args.cost_input_type == 'ss':
        #TODO: Change. Currently, the assumption is that if cost_input_type is ss, the expert dataset comes from reference motion so only has s, s' in dataset.
        expert_state, expert_next_state = get_db_mjrl(args.expert_data, expert=True, imitate_amp=imitate_amp)
    else:
        expert_state, expert_action, expert_next_state = get_db_mjrl(args.expert_data, imitate_amp=imitate_amp)
        
    if not args.mlp_cost:
        if args.cost_input_type == 'ss':
            cost_function = RBFLinearCost(torch.cat([expert_state, expert_next_state], dim=1),
                                        feature_dim=args.cost_feature_dim, \
                                        input_type=args.cost_input_type, bw_quantile=args.bw_quantile,
                                        lambda_b=args.lambda_b, seed=args.seed)
        elif args.cost_input_type == 'sa':
            cost_function = RBFLinearCost(torch.cat([expert_state, expert_action], dim=1),
                                        feature_dim=args.cost_feature_dim, \
                                        input_type=args.cost_input_type, bw_quantile=args.bw_quantile,
                                        lambda_b=args.lambda_b, seed=args.seed)
    else:
        if args.cost_input_type == 'ss':
            cost_function = MLPCost(torch.cat([expert_state, expert_next_state], dim=1),
                                    hidden_dims=[512, 512, 512, 512],
                                    feature_dim=args.cost_feature_dim,
                                    input_type=args.cost_input_type,
                                    bw_quantile=args.bw_quantile,
                                    lambda_b=args.lambda_b,
                                    seed=args.seed)
        elif args.cost_input_type == 'sa':
            cost_function = MLPCost(torch.cat([expert_state, expert_action], dim=1),
                                    feature_dim=args.cost_feature_dim,
                                    input_type=args.cost_input_type,
                                    bw_quantile=args.bw_quantile,
                                    lambda_b=args.lambda_b,
                                    seed=args.seed)

    # ==============Init Agents==============#
    policy = MLP(state_size, action_size, hidden_sizes=tuple(args.actor_model_hidden), seed=args.seed,
                 init_log_std=args.policy_init_log, min_log_std=args.policy_min_log)
    baseline = MLPBaseline(state_size, reg_coef=args.vf_reg_coef, batch_size=args.vf_batch_size, \
                           hidden_sizes=tuple(args.critic_model_hidden), epochs=args.vf_iters, learn_rate=args.vf_lr)
    start_training_time = time.time()


    # ==============BC Warmstart==============#
    if args.bc_epochs > 0:
        loggers['milo'].info(f">>>>> BC Warmstart for {args.bc_epochs} epochs")
        offline_paths = get_paths_mjrl(args.offline_data)
        bc_agent = BC(offline_paths, policy=policy, epochs=args.bc_epochs, batch_size=64, lr=1e-3)
        bc_agent.train()
        # Reinit Policy Std
        policy_params = policy.get_param_values()
        policy_params[-1 * action_size:] = args.policy_init_log
        policy.set_param_values(policy_params, set_new=True, set_old=True)

    # ============== Policy Gradient Init =============
    if args.cost_input_type == 'ss':
        expert_paths = get_paths_mjrl(args.expert_data, expert=True)
    elif args.cost_input_type == 'sa':
        expert_paths = get_paths_mjrl(args.expert_data)
    bc_reg_args = {'flag': args.do_bc_reg, 'reg_coeff': args.bc_reg_coeff, 'expert_paths': expert_paths[0]}
    if args.planner == 'trpo':
        cg_args = {'iters': args.cg_iter, 'damping': args.cg_damping}
        planner_agent = NPG(mb_env, policy, baseline, normalized_step_size=args.kl_dist,
                            hvp_sample_frac=args.hvp_sample_frac, seed=args.seed, FIM_invert_args=cg_args,
                            bc_args=bc_reg_args, save_logs=True)
    else:
        raise NotImplementedError('Chosen Planner not yet supported')

    # ==============================================
    # ============== MAIN LOOP START ===============
    # ==============================================
    n_iter = 0
    best_policy_score = -float('inf')
    while n_iter < args.n_iter:
        loggers['milo'].info(f"{'=' * 10} Main Episode {n_iter + 1} {'=' * 10}")
        loggers['milo'].info("Evaluating....")
        scores, mmds = evaluate(n_iter, loggers['milo'], writers['milo'], args, inf_env, planner_agent.policy,
                                cost_function, num_traj=args.num_eval_traj, imitate_amp=imitate_amp)

        save_and_plot(n_iter, args, milo_directories, scores, mmds)

        if scores['greedy'] > best_policy_score:
            best_policy_score = scores['greedy']
            save_checkpoint(milo_directories, planner_agent, cost_function, 'best', agent_type=args.planner)
        if (n_iter + 1) % args.save_iter == 0:
            save_checkpoint(milo_directories, planner_agent, cost_function, n_iter + 1, agent_type=args.planner)

        # =============== DO PG STEPS =================
        loggers['milo'].info(f'===PG Planning Start')
        best_baseline_optim, best_baseline = None, None
        curr_max_reward, curr_min_vloss = -float('inf'), float('inf')
        for i in range(args.pg_iter):
            reward_kwargs = dict(reward_func=cost_function, ensemble=dynamic_ensemble, device=device)
            planner_args = dict(N=args.samples_per_step, env=mb_env, sample_mode='model_based', \
                                gamma=args.gamma, gae_lambda=args.gae_lambda, num_cpu=args.num_cpu, \
                                eval_mode=args.eval_mode, verbose=False,
                                reward_kwargs=reward_kwargs)  # TODO: FIX EVAL MODE
            r_mean, r_std, r_min, r_max, _, infos = planner_agent.train_step(**planner_args)

            # Baseline Heuristic
            if infos['vf_loss_end'] < curr_min_vloss:
                curr_min_vloss = infos['vf_loss_end']
                best_baseline = planner_agent.baseline.model.state_dict()
                best_baseline_optim = planner_agent.baseline.optimizer.state_dict()
            # Stderr Logging
            reward_mean = np.array(infos['reward']).mean()
            int_mean = np.array(infos['int']).mean()
            ext_mean = np.array(infos['ext']).mean()
            len_mean = np.array(infos['ep_len']).mean()
            loggers['milo'].info(f"alpha: {infos['alpha']}")
            loggers['milo'].info(f"kl_dist: {infos['kl_dist']}")
            loggers['milo'].info(f"npg_grad: {infos['npg_grad']}")
            loggers['milo'].info(f"vpg_grad: {infos['vpg_grad']}")
            loggers['milo'].info(f"Advantages: {infos['advantages']}")
            loggers['milo'].info(f"Old Dist Info: {infos['old_dist_info']}")
            loggers['milo'].info(f"New Dist Info: {infos['new_dist_info']}")
            loggers['milo'].info(f"Likelihood Ratio: {infos['lr']}")
            loggers['milo'].info(f"Surr before: {infos['surr_before']}")
            loggers['milo'].info(f"Surr after: {infos['surr_after']}")

            loggers['milo'].info(f"Running Score: {infos['running_score']}")
            loggers['milo'].info(f"Surr improvement: {infos['surr_improvement']}")

            ground_truth_mean = np.array(infos['ground_truth_reward']).mean()
            loggers['milo'].info(f'Model MMD: {infos["mb_mmd"]}')
            loggers['milo'].info(f'Bonus MMD: {infos["bonus_mmd"]}')
            loggers['milo'].info(f'Model Ground Truth Reward: {ground_truth_mean}')
            loggers['milo'].info('PG Iteration {} reward | int | ext | ep_len ---- {:.2f} | {:.2f} | {:.2f} | {:.2f}' \
                                 .format(i + 1, reward_mean, int_mean, ext_mean, len_mean))
            save_sim_rewards((n_iter+1)*(i+1)*args.samples_per_step, milo_directories, int_mean, ext_mean, len_mean)

            # Tensorboard Logging
            step_count = n_iter * args.pg_iter + i
            writers['milo'].add_scalar('data/reward_mean', reward_mean, step_count)
            writers['milo'].add_scalar('data/ext_reward_mean', ext_mean, step_count)
            writers['milo'].add_scalar('data/int_reward_mean', int_mean, step_count)
            writers['milo'].add_scalar('data/ep_len_mean', len_mean, step_count)
            writers['milo'].add_scalar('data/true_reward_mean', ground_truth_mean, step_count)
            writers['milo'].add_scalar('data/value_loss', infos['vf_loss_end'], step_count)
            writers['milo'].add_scalar('data/mb_mmd', infos['mb_mmd'], step_count)
            writers['milo'].add_scalar('data/bonus_mmd', infos['bonus_mmd'], step_count)
        planner_agent.baseline.model.load_state_dict(best_baseline)
        planner_agent.baseline.optimizer.load_state_dict(best_baseline_optim)
        n_iter += 1
    print(f"Total time to train: {time.time() - start_training_time}")
    loggers['milo'].info(f"Total time to train: {time.time() - start_training_time}")
    loggers['milo'].info(f"Total time overall: {time.time() - all_start}")


if __name__ == '__main__':
    main()