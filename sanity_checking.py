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

from deepmimic.util.arg_parser import ArgParser
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.behavior_cloning import BC
from mjrl.algos.npg_cg import NPG
import time

def test_model():
    model_start = time.time()
    args = get_args()
    milo_directories, dynamic_directories, load_dynamics, loggers, writers, device = setup(args, ask_prompt=False)
    
    arg_parser = ArgParser()
    arg_parser.load_file(args.deepmimic)
    imitate_amp = arg_parser.parse_string('scene') == 'imitate_amp'
    env = DeepMimicEnv(['--arg_file', args.deepmimic], False)
    
    state_size = env.get_state_size(0)
    action_size = env.get_action_size(0)
    
    # dataset loading
    offline_state, offline_action, offline_next_state = get_db_mjrl(args.offline_data, imitate_amp=imitate_amp)
    offline_dataset = AmpDataset(offline_state, offline_action, offline_next_state, device)
    
    # Validate dataset for training dynamics model
    if args.validate:
        validate_state, validate_action, validate_next_state = get_db_mjrl(args.validate_data, imitate_amp=imitate_amp)
        validate_dataset = AmpDataset(validate_state, validate_action, validate_next_state, device)
    else:
        validate_dataset = None

    # Create dynamics Model
    if args.dynamic_optim == 'sgd':
        optim_args = {'optim': 'sgd', 'lr': args.dynamic_lr, 'momentum': args.dynamic_momentum}
    elif args.dynamic_optim == 'adam':
        optim_args = {'optim': 'adam', 'lr': args.dynamic_lr, 'eps': args.dynamic_eps}
    else:
        assert False, "Choose a valid optimizer"

    # Train dynamics model
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
    
    # get model-based env
    reset_args = dict(
        custom_time=args.custom_time,
        time_min=args.time_min,
        time_max=args.time_max,
        resolve=not args.no_resolve,
        noise_bef_rot=args.noise_bef_rot,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        radian=args.radian,
        rot_vel_w_pose=args.rot_vel_w_pose,
        vel_noise=args.vel_noise,
        interp=args.interp,
        knee_rot=args.knee_rot
    )
    mb_env = gym.make('simenv-v0', deepmimic_args=args.deepmimic, dynamic_ensemble=dynamic_ensemble, reset_args=reset_args)
    real_env = gym.make('deepmimic-v0', deepmimic_args=args.deepmimic, reset_args=reset_args)
    
    # ====== now do model-based RL ======
    policy = MLP(state_size,
                          action_size,
                          hidden_sizes=tuple(args.actor_model_hidden),
                          seed=args.seed,
                          init_log_std=args.policy_init_log,
                          min_log_std=args.policy_min_log)
    
    baseline = MLPBaseline(state_size,
                           reg_coef=args.vf_reg_coef,
                           batch_size=args.vf_batch_size,
                           hidden_sizes=tuple(args.critic_model_hidden),
                           epochs=args.vf_iters,
                           learn_rate=args.vf_lr)
    
    # bc warmstart
    if args.bc_epochs > 0:
        loggers['milo'].info(f">>>>> BC Warmstart for {args.bc_epochs} epochs")
        offline_paths = get_paths_mjrl(args.offline_data)
        bc_agent = BC(offline_paths, policy=policy, epochs=args.bc_epochs, batch_size=64, lr=1e-3)
        bc_agent.train()
        # Reinit Policy Std
        policy_params = policy.get_param_values()
        policy_params[-1 * action_size:] = args.policy_init_log
        policy.set_param_values(policy_params, set_new=True, set_old=True)
        
    # agent + expert dataset init
    if args.cost_input_type == 'ss':
        expert_paths = get_paths_mjrl(args.expert_data, expert=True)
    elif args.cost_input_type == 'sa':
        expert_paths = get_paths_mjrl(args.expert_data)
        
    bc_reg_args = {'flag': args.do_bc_reg, 'reg_coeff': args.bc_reg_coeff, 'expert_paths': expert_paths[0]}
    
    if args.planner == 'trpo':
        cg_args = {'iters': args.cg_iter, 'damping': args.cg_damping}
        planner_agent = NPG(mb_env,
                            policy,
                            baseline,
                            normalized_step_size=args.kl_dist,
                            hvp_sample_frac=args.hvp_sample_frac,
                            seed=args.seed,
                            FIM_invert_args=cg_args,
                            bc_args=bc_reg_args,
                            save_logs=True)
    else:
        raise NotImplementedError("haven't implemented any other planner yet")
    
    # training + evaluation
    n_iter = 0
    best_policy_score = -float('inf')
    while n_iter < args.n_iter:
        loggers['milo'].info(f"{'=' * 10} Main Episode {n_iter + 1} {'=' * 10}")
        loggers['milo'].info("Evaluating....")
        
        # TODO write evaluation + saving part
        scores, _ = evaluate(n_iter,
                             loggers['milo'],
                             writers['milo'],
                             args,
                             real_env,
                             planner_agent.policy,
                             None,
                             num_traj=args.num_eval_traj,
                             imitate_amp=imitate_amp)
    
        # TODO write training part
        loggers['milo'].info(f'====== PG Planning Start ======')
        best_baseline_optim, best_baseline = None, None
        curr_max_reward, curr_min_vloss = -float('inf'), float('inf')
        for i in range(args.pg_iter):
            reward_kwargs = dict(ensemble=dynamic_ensemble, device=device) # reward_func = cost_function? TODO define GAIL cost and go from there or use actual rewards
            planner_args = dict(N=args.samples_per_step,
                                env=mb_env,
                                sample_mode='model_based',
                                gamma=args.gamma,
                                gae_lambda=args.gae_lambda,
                                num_cpu=args.num_cpu,
                                eval_mode=args.eval_mode,
                                verbose=False,
                                reward_kwargs=reward_kwargs)  # TODO: fix to get actual rewards from env
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
        
        n_iter += 1