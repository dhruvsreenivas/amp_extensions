## Source Code for Model-based Imitation Learning from Offline data (MILO)
The original repo can be found [here](https://github.com/jdchang1/milo) and contains documentation for setup and experimentation running. 

## Setup
Make sure to follow the setup instructions [here](../setup.md) and install this package with 

```pip install -e .```

## Changes made to the original code
Additional arguments were added```arguments.py``` for  resetting the character in ```DeepMimicCore``` and to ```utils.py``` in order to adapt the code to work with the AMP framework. 


`sampler.py` contains utility functions used for sampling samples from gym environments using the MJRL policies. The two sections that follow describe the other two files. 

`dynamics.py` was heavily inspired by the original dynamics code in MILO. A slight adaption was made to allow for DenseNet-style MLPs. 

```collect_data.py, collect_expert.py, sampler.py``` are three new files added and ```run_amp_humanoid3d_spinkick_args.txt``` is an example of the argument file used to create ```DeepMimicEnv```. 

## Collecting the offline datasset for MILO from the AMP framework
To generate the offline dataset, there are a few steps. 

#### Collect checkpoints from AMP framework
For a given motion (and goal if applicable), use the AMP framework to train a policy using the various ```train``` argument files in the [```args```](../deepmimic/deepmimic/args) directory. For more information on how to train the policy, see the [`readme`](deepmimic/README.md) for `DeepMimic` or look at the original `DeepMimic` repository. For example, in the [DeepMimic](../deepmimic/deepmimic) directory, you can train a spinkick policy by running
  ```python mpi_run.py --arg_file args/train_humanoid3d_spinkick_args.txt --num_workers 16```.
  
Make sure to uncomment the line for ```int_output_path``` in the train argument file in order to enable checkpoint saving. To control how often to save checkpoints, go into the agent file specified in the argument file and edit ```IntOutputIters```. The agent should be in the [```agent```](../deepmimic/deepmimic/data/agents) directory. 

If the default parameters were used for training and recording checkpoints, then the checkpoints should have a name like ```agent0_int_model000000<checkpoint_number>.ckpt```. There 
should be ```.index``` and ```.data``` files. To load a model, we simply specify ```agent0_int_model000000<checkpoint_number>.ckpt``` to TensorFlow and it will load. 
#### Record stats for each checkpoint
The next step is to record statistics for these policies. Specifically, we rollout many trajectories using each checkpoint model and save
the average return (or DTW cost if the scene is ```imitate_amp```) and average length of a trajectory using that checkpoint. 

To do this, you need to first decide on the following parameters.
- ```record_stats```: This argument needs to be included to enable recording stats
- ```world_arg_file```: This argument file decides the motion, character, horizon, etc. See ```run_amp_humanoid3d_spinkikck_args.txt``` for an example of how to modify the existing argument files in [```../deepmimic/deepmimic/args```](../../deepmimic/deepmimic/args). 
- ```stat_repetitions```: Number of trajectories to collect for each checkpoint
- ```stat_save_directory```: Directory to store statistics.
- ```stat_name```:  Name of file where statistics will be saved. 
- ```lower_model```: This is the checkpoint number of the first model to record statistics for
- ```upper_model```: Last model to record statistics for. 
- ```checkpoint_save_iters```: How often checkpoints were saved. If you did not change ```IntOutputIters``` in the ```agent``` file, then this should be default 200.
- ```model_folder```: This should be the path to the folder containing the checkpoints. should be able to be used. Check to make sure this path is correct. 
- ```model_prefix```: If the AMP code was followed without any modifications, the default should be able to be used. 

The other parameters that are important are the reset-specific parameters. See the ```trajectory_reset_parameters``` in the```get_args```function for more details. The default behavior is to reset the character to a random time in the motion (i.e., in the time range `[0, <end of motion>]`) without adding any noise. 

An example command (run in the [child](milo) directory) would be

`python collect_data.py --record_stats --stat_repetitons 50 --stat_save_directory ../data/stats --dataset_name offline --lower_model 0 --upper_model 36000`

This collects 50 trajectories from each of the checkpoints numbered 0, 200, ..., 36200 and saves the statistics in ```../data/stats/stats.pt```. 
For each trajectory, the character is reset to a random time in the motion with no noise added. 

#### Collect trajectories
With the statistics, you can collect data. First, it is recommended to go through the statistics and look through the 
average score (reward or DTW cost) and lengths of the trajectories sampled for each checkpoint. 

Like when recording stats, there are a few key parameters. The parameters that should remain the same are ```world_arg_file, checkpoint_save_iters, model_folder, model_prefix```, and all the reset parameters. 
The collection-specific parameters to pay attention to are:
- ```dataset_directory```: Directory where the data will be saved.
- ```dataset_name```: Name of dataset. 
- ```stat_path```: Path to stats that were collected in the previous step. 
- ```lower_model```: This is the checkpoint number of the first model to collect trajectories for. This doesn't need to match what was used when recording stats. 
- ```upper_model```: Last model to collect data for. 
- ```use_length_boundaries, lower_limits, repetitions``` - see below

The final three parameters ```use_length_boundaries, lower_limits, repetitions``` are the key for collecting the data. 

The current implementation decides the number of trajectories to sample for each model in the checkpoint range ```[lower_model, upper_model]```. For a given model x, we first lookup the average score or the average length depending on the ```use_length_boundaries``` flag. For example, if it is false, we use the average score. Then, 
we find the index of the largest value in ```lower_limits``` less than the score. Finally, we use this index in ```repetitions``` to determine the number of trajectories to sample for ```x```. The only restriction is that ```len(lower_limits)==len(repetitions)```. 

An example command (run in the [child](milo) directory)  would be

`python collect_data.py --stat_path ../data/stats/stat.pt --lower_model 4200 --upper_model 5400 --repetitions [1000 1800 200]`

This uses average score (DTW cost for ```imitate_amp``` scenes and reward otherwise) to compute the number of trajectories for checkpoints in the range [4200, 5400]. The lower limits are the default ```[0, 100, 200]``` and the repetitions are `[1000, 1800,200]`. Everything else is default like the resetting or dataset path/name. 

The dataset stored is a list of dictionary where each dictionary corresponds to a trajectory. The dicationary has the form
`{'episode':(states, actions, scores): <score_name>:score}`. If the scene is `imitate_amp`, the `<score_name>` is `dtw_cost` and the scores are the backtracked DTW cost. If the scene is a `imitate` type which means it's using DeepMimic rather than AMP, then `<score_name>` is `ep_rew` and the scores are the rewards at each timestep.
## Collecting the expert dataset for MILO from the AMP framework 
Collecting data from the expert for DeepMimic is much simpler. There are two parameters that are important 
- ```world_arg_file```: This controls the scene that is loaded. 
- ```num_traj```: This represents the number of trajectories to sample beyond the 2 in the motion file. For most of the motion clips, the keyframes are specified at 60Hz so we can extract 2 trajectories without any need for interpolation. Thus, there will always be at least 2 trajectories in the output expert dataset. 

An example command (run in the [child](milo) directory)  would be

`python collect_expert --world_arg_file run_amp_humanoid3d_spinkick_args.txt --num_traj 498`

This will save 500 expert trajectories in ```../data/expert.pt```

At a high level, the way each trajectory is sampled is first a random time is sampled between 0 and the motion length. Then, the kinematic character (which contains information about the reference motion)
is loaded onto the simulated character. If ```no_resolve==False```, then we resolve any ground intersections which is the default behavior. Then, the character is rolled out by stepping forward in time (typically 1/30s) and re-syncing the kinematic and simulated character before recording the state. 


