 # gym-simenv

This gym environment serves as a wrapper around the dynamics ensemble for MILO. This dynamics ensemble attempts to model the dynamics of a specific scene in the AMP framewoerk (i.e., spinkick). 

## Setup
Make sure to follow the setup instructions [here](../setup.md) and install this package with 

```pip install -e .```
## Creating an environment
```
import gym 
import gym_simenv
env = gym.make('simenv-v0', deepmimic_args=<path to arg file>, dynamics_ensemble=<dynamics_ensemble, reset_args=<reset_args>)
```
## Setting the scene/argument file
Make sure that the dynamics ensemble is modeling the dynamics of the scene specified by ```deepmimic_args```. To set this correctly, please read the
section `Setting up the argument file` in the [`README`](../README.md). Make sure the horizon is set correctly and the correct scene (i.e. imitation + goal)
are loaded. 

Additionally, the data used to collect the dynamics ensemble should be from the scene specified by ```deepmimic_args``` and the ```reset_args``` should match as well. 

## ```init```
gym-simenv inherits from both ```gym.Env``` and ```gym.utils.EzPickle```. The latter is to allow pickling of the arguments so our environment can be compatible with multiprocessing libraries. 
## ```step```
gym-simmenv uses the dynamics ensemble to step to the next state using the current selected model in the ensemble. To determine termination, there are three possible conditions: horizon met, ground collision, velocity threshold exceeded. Unlike typical gym environments, the reward will alwyas be returned here. The main purpose of this environment is to use in the ```milo``` algorithm so the reward returned here is not important. 


## ```reset```
gym-simenv uses ```DeepMimicEnv``` to reset the character based on the reset parameters. For more information on how to reset the character, see the [`DeepMimic README.md`](../deepmimic/README.md). Additionally, the current model used for stepping in the ensemble is updated to different model. 
