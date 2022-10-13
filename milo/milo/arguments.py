import argparse


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #==============General Arguments==============#
    
    parser.add_argument("--experiment_name", type=str, default="Experiment")
    parser.add_argument('--root_path', type=str, help='Root dir to save outputs', default='./experiments')
    parser.add_argument('--deepmimic', type=str, help='arguments for creating DeepMimicEnv', default='run_amp_humanoid3d_spinkick_args.txt') 

    parser.add_argument('--offline_data', type=str, help='path to offline data', default='./data/offline.pt')

    parser.add_argument('--expert_data', type=str, help='path to expert data', default='./data/expert.pt')
    parser.add_argument('--validate_data', type=str, help='path to validation data', default='./data/validate.pt')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--seed', type=int, help='seed', default=100)
    parser.add_argument('--dynamic_id', type=int, help='dynamic experiment id', default=0)
    parser.add_argument('--milo_id', type=int, help='milo experiment id', default=0)
    parser.add_argument('--num_cpu', type=int, help='number of processes used for inference', default=4)

    #==============Reset arguments for resetting character in DeepMimic and Dynamics Model==============#
    
    parser.add_argument('--custom_time', action='store_true',
                        help='When true, resets to random time between time_min and time_max. Otherwise, resets to time between 0 and end of motion.')
    parser.add_argument('--time_min', default=0, type=float, help='lower bound on reset time')
    parser.add_argument('--time_max', default=0, type=float, help='upper bound on reset time')
    parser.add_argument('--no_resolve', action='store_true',
                        help='Resolve flag tells DeepMimicCore whether to resolve ground intersections when resolving. In most cases, should resolve so no_resolve==False')
    parser.add_argument('--noise_bef_rot', action='store_true',
                        help='There are two methods used to add randomness to state when resetting. This flag controls the order. Check DeepMimicCore KinCharacter.cpp for info on the two methods')
    parser.add_argument('--noise_min', type=float, default=0, help='Lower bound on amount of noise added to state')
    parser.add_argument('--noise_max', type=float, default=0, help='Upper bound on amount of noise added to state')
    parser.add_argument('--radian', type=float, default=0,
                        help='Amount of allowed rotation (in both positive and negative direction) when adding noise')
    parser.add_argument('--rot_vel_w_pose', action='store_true',
                        help='When true, any random roation applied during reset to pose is also applied to velocity')
    parser.add_argument('--vel_noise', action='store_true', help='When true, random noise added to velocity as well')
    parser.add_argument('--interp', type=float, default=1,
                        help='A float between [0,1] indicating how to initialize velocity during reset. 1 means to reset to expert. 0 is zero velocity')
    parser.add_argument('--knee_rot', action='store_true', help='When true, random noise added to knees in state')

    #==============Dynamic Model arguments==============#
    
    #=========general args=========#
    parser.add_argument('--dynamic_epochs', type=int, help='number of epochs to train dynamic model', default=500)
    parser.add_argument('--dynamic_grad_clip', type=float, help='Max Gradient Norm', default=1.0)
    parser.add_argument('--dynamic_batch_size', type=int, help='Batch size for training dynamics model', default=256)
    parser.add_argument('--dynamic_hidden_sizes', nargs='+', help='width of layers used for dynamics model', type=int, default=[1024, 1024])
    parser.add_argument('--dynamic_dense_connect', help='bool indicating whether to create MLP in densenet fashion', action='store_true')
    parser.add_argument('--dynamic_num_models', type=int, default=4)
    
    #=========Optimizer specific args=========#
    parser.add_argument('--dynamic_optim', type=str, help='Optimizer to use [sgd, adam]', default='adam')
    parser.add_argument('--dynamic_lr', type=float, help='lr for training dynamic model', default=1e-3)
    parser.add_argument('--dynamic_momentum', type=float, help='momentum used for sagd optimizer', default=0.9)
    parser.add_argument('--dynamic_eps', type=float, help='eps used for adam optimizer', default=1e-8)
    parser.add_argument('--dynamic_transform', action='store_true')
    
    #=========Save parameters=========#
    parser.add_argument('--dynamic_save_models', action='store_true')
    parser.add_argument('--dynamic_save_checkpoints',  help='Option to also save intermediate model checkpoints (currently hardcoded to every 100 epochs', action='store_true')
    parser.add_argument('--load_dynamics', action='store_true')
    parser.add_argument('--dynamic_checkpoint', help='Path to checkpoint. This will be loaded if not None', type=str, default=None)
    
    #==============Cost arguments==============#
    parser.add_argument('--mlp_cost', action='store_true', help='whether to train MLP phi(s, a) or linear phi(s, a)')
    parser.add_argument('--cost_feature_dim', help='Dimension of rff features', type=int, default=512)
    parser.add_argument('--cost_input_type', type=str, default='ss')
    parser.add_argument('--bw_quantile', type=float, help='Quantile when fitting bandwidth', default=0.2)
    parser.add_argument('--lambda_b', type=float, help='Bonus/Penalty weighting param', default=0.1)
    parser.add_argument('--cost_lr', type=float,
                        help='0.0 is exact update, otherwise learning rate', default=0.0)
    parser.add_argument('--update_type', type=str,
                        help='exact, geometric, decay, decay_sqrt, ficticious', default='exact')

    #==============MILO args==============#
    #======PG args======#
    parser.add_argument('--eval_mode', action='store_true', help='boolean used to determine whether to use mean output of gaussian policy when sampling policy')
    parser.add_argument('--planner', type=str, help='pg alg to use (trpo, ppo)', default='trpo')
    parser.add_argument('--actor_model_hidden', type=int, nargs='+', help='hidden dims for actor', default=[32, 32])
    parser.add_argument('--policy_init_log', type=float, help='policy init log', default=-0.25)
    parser.add_argument('--policy_min_log', type=float, help='policy min log', default=-2.0)
    parser.add_argument('--vf_reg_coef', type=float, help='baseline regularization coeff', default=1e-3)
    parser.add_argument('--vf_batch_size', type=int, help='Critic batch size', default=64)
    parser.add_argument('--critic_model_hidden', type=int, nargs='+', help='hidden dims for critic', default=[128, 128])
    parser.add_argument('--vf_iters', type=int, help='Number of value optim steps', default=2)
    parser.add_argument('--vf_lr', type=float, help='Value lr', default=1e-3)
    parser.add_argument('--samples_per_step', type=int,
                        help='Number of mb samples per pg step', default=40000)
    parser.add_argument('--gamma', type=float,
                        help='discount factor for rewards (default: 0.99)', default=0.995)
    parser.add_argument('--gae_lambda', type=float,
                        help='gae lambda val', default=0.97)
    #======BC warmup args======#
    parser.add_argument('--bc_epochs', type=int,
                        help='Number of BC epochs', default=3)

    #======BC regularization Arguments======#
    parser.add_argument('--do_bc_reg', action='store_true', help="Add bc regularization to policy gradient. Default to false since DeepMimic uses (s,s')")
    parser.add_argument('--bc_reg_coeff', type=float, help='Regularization coefficient for policy gradient',
                        default=0.1)

    #======TRPO======#
    parser.add_argument('--cg_iter', type=int, help='Number of CG iterations', default=10)
    parser.add_argument('--cg_damping', type=float, help='CG damping coefficient', default=1e-4)
    parser.add_argument('--kl_dist', type=float, help='Trust region', default=0.05)
    parser.add_argument('--hvp_sample_frac', type=float, help='Fraction of samples for FIM', default=1.0)

    # ======General Algorithm Arguments======#
    parser.add_argument('--n_iter', type=int, help='Number of offline IL iterations to run', default=300)
    parser.add_argument('--pg_iter', type=int, help='Number of pg steps', default=5)
    parser.add_argument('--num_eval_traj', type=int, help='Number of trajectories to sample in real env for evaluation each step of IL', default=50)
    parser.add_argument('--save_iter', type=int, help='Interval to Save checkpoints', default=10)
    args = parser.parse_args()
    return args
