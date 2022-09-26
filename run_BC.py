from deepmimic.env.deepmimic_env import DeepMimicEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.algos.behavior_cloning import BC
import torch 
import argparse

from milo.utils import *
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="BC arguments")
    parser.add_argument("--env_args", type=str, default='amp_args.txt', help='args for creating the core env')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size',type=int, default=128, help='size of both hidden layers')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--states_stored_as_one', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data/offline.pt')
    parser.add_argument('--save_name', type=str, default='bc')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    reset_args = dict(custom_time=False, time_min=0, time_max=0,
                      resolve=True,
                      noise_bef_rot=False, noise_min=0, noise_max=0,
                      radian=0, rot_vel_w_pose=False, vel_noise=False,
                      interp=1, knee_rot=False)

    print(f'hidden size: {args.hidden_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Data path: {args.data_path}')
    print(f'lr: {args.lr}')
    env = DeepMimicEnv(['--arg_file', args.env_args], False)
    print('CREATED ENV...CREATING POLICY')
    policy = MLP(env.get_state_size(0), env.get_action_size(0), hidden_sizes=(args.hidden_size, args.hidden_size), seed=args.seed)
    print("CREATED POLICY...LOADING DATA")
    offline_paths = get_paths_mjrl(args.data_path,'all')

    print('LOADED DATA...CREATING BC AGENT')
    bc_agent = BC(offline_paths, policy=policy, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print('CREATED BC AGENT...TRAINING')
    #sys.exit(0)
    logs =bc_agent.train()
    print("TRAINED...SAVING")
    file_name = args.data_path
    if '/' in args.data_path:
        file_name = file_name[file_name.rfind('/')+1:]
    
    torch.save(policy, f'{args.save_name}.pt')
    torch.save(logs, f'{args.save_name}_logs.pt')


    
