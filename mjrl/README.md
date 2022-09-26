# RL for MuJoCo

This packag contains implementations of various RL algorithms for continuous control tasks simulated with [MuJoCo.](http://www.mujoco.org/) The original repo can be found [here](https://github.com/aravindr93/mjrl). 
## Setup
Make sure to follow the setup instructions [here](../setup.md) and install this package with 

```pip install -e .```

## Log of changes
While MJRL was originally meant for use with MuJoCo envs, we were able to use the implementations of the RL algorithms with ```gym-deepmimic``` with some minor changes. 

Shown below is a list of changes made to the MJRL repo. Most of these changes are minor and follow the same changes made by MILO. 
- [```mjrl/__init__.py```](mjrl/__init__.py) 
    - Since we are not using mujoco envs, the ```import mjrl.envs``` statement was commented out. 
- [```mjrl/policies/gaussian_mlp.py```](mjrl/policies/gaussian_mlp.py) 
    - In ```__init__```, instead of taking in ```env_spec```, the function takes in the ```input_size``` and ```output_size```. 
    - The new paramaeter ```eps``` represents the probability of the policy sampling a random action.
- [```mjrl/baselines/mlp_baseline.py```](mjrl/baselines/mlp_baseline.py) 
    - If returning errors in ```fit```, the function also returns the epoch losses. 
- [```mjrl/algos/batch_reinforce.py```](mjrl/algos/batch_reinforce.py) 
    - In ```__init__```, the ```env_spec``` parameter was removed. 
    - Like in the MILO repository, ```train_step``` is adapted to sample points when ```sample_mode=='model_based'```. 
