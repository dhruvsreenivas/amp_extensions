 # gym-deepmimic

This gym environment primarily serves as a wrapper around a [`DeepMimicEnv`](../deepmimic/deepmimic/env/deepmimic_env.py) object so that `DeepMimicCore`
functions can be used in an OpenAi-gym environment. 


## Setup
Make sure to follow the setup instructions [here](../setup.md) and install this package with 

```pip install -e .```

## Creating an environment
```
import gym 
import gym_deepmimic
env = gym.make('simenv-v0', deepmimic_args=<path to arg file>, reset_args=<reset_args>)
```

## Use cases
This environment can be used to train RL algorithms using the characters, controllers, worlds, etc., in `DeepMimicCore` by loading a `DeepMimicEnv`
object which itself contains the SWIG wrapper for `DeepMimicCore`. This has been
tested primarily with the `imitate_amp` scene with the humanoid character (see [`../run_amp_humanoid3d_spinkick_args.txt`](../run_amp_humanoid3d_spinkick_args.txt) for an example).
It should be able to work with DeepMimic-types scenes as well as scenes that have goal tasks, however these have not been tested. 

Note that this environment does not connect to the RL side of the AMP framework which means the reward there is no style returned for AMP scenes by the environment. To do this, a few modifications should be made with the primary one being to 
replace the `DeepMimicEnv` object with an `RLWorld` object (see [`rl_world.py`](../deepmimic/deepmimic/learning/rl_world.py))

## ```__init__```
The key steps in the constructor are to initialize the ```DeepMimicEnv``` object and to set the substep timestep and total number of timesteps taken in one update. Other important variables such as the state, action, and goal space as well as the reset parameters are defined. 
## ```step```
#### Updating to the next state
The code for updating to the next step is based on `update_world` in [`DeepMimic.py`](../deepmimic/deepmimic/DeepMimic.py). In the AMP framework, [`DeepMimic_Optimizer`](../deepmimic/deepmimic/DeepMimic_Optimizer.py) updates by calling `update_world` with a timestep of 1/60.  The `update_world` function further divides this by 10 (using `world.env.get_num_update_substeps()` which defaults to 10) so `update_world` updates the world in `DeepMimicCore` 10 times with timestep 1/600. However, the `DeepMimicCore` further subdivides this into 2 and does 2 sub-updates with timestep 1/1200. This is why the world in AMP is said to update at a frequency of 1.2KHz.

Since the policy runs at a frequency of 30Hz and the `step` function is supposed to get us the next state after applying the input action, the code in `step` essentially does 2 `update_world` calls as `self.total_substeps` defaults to 20 and `self.substep_timestep` defaults to 1/600 like in `update_world`. 

#### Checking validiity
Like in `update_world`, we also check if the episode is 'valid'. That is, if `DeepMimicCore` determines that the velocity of the character has exceeded a certain threshold, the episode is considered 'invalid' and is not used. We do the same in `step`.

#### Recording the state
If the scene is imitation-only (either DeepMimic or AMP), then only the state will be returned. However, if there is a goal, the state and goal will be concatenated together before returning as policies in these types of scenes are (s,g) dependent. 

## ```reset```
Using the reset parameters saved in `__init__`, the environment is reset. The default reset behavior is to reset the character to a random point in the reference motion without adding additional noise. For more information on how to reset the character, see the [`DeepMimic README.md`](../deepmimic/README.md)

## `render`
The rendering code is again based on `DeepMimic.py`. 

#### Visualization in `DeepMimic.py`
In `DeepMimic.py`, the policy and world are stored as global variables and GLUT is used for visualization. 

`DeepMimic.py` uses `glutMainLoop` which enters the GLUT event processing loop. This routine never returns so the user can no longer run code and can only interact with the program using GLUT events (i.e., keyboard or mouse events) which call user-defined callback functions. In addition, whenever GLUT believes the screen needs to be necessary, it calls the `draw` callback function registered with `glutDisplayFunc`. 

The policy is visualized at 60fps (i.e., 2 frames per action) by defining a timer event that goes off every 1/60 seconds. This event will call the `animate` function in `DeepMimic.py` which updates the world and when necessary, queries the policy for a new action. This is why the world and policy are set as global variables so the program can access them in these callback functions even in the GLUT event processing loop. 

This approach to visualization using `glutMainLoop` is a problem for `gym_deepmimic`. The environment itself does not have access to the policy to query actions and also is interactive by nature so entering the GLUT event processing loop makes it so the user cannot use the gym environment instead.

#### Visualization in `gym-deepmimic`
In `gym-deepmimic`, GLUT is initialized and setup in `__init__` in the same way as in `DeepMimic.py`. However, instead of registering callback functions for events such as keyboard or mouse press, the only callback functions defined are for drawing. Whenever `render` is called, `glutMainLoopEvent` instead of `glutMainLoop` is called which executes one loop of the GLUT event processing loop, updating the screen. This enables the user to reset and step as many times as they want and then render the current state whenever desired. Since `gym-deepmimic` does not enter the GLUT event processing loop like in `DeepMimic.py`, GLUT is not able to respond to keyboard events, mouse events, etc. that require constant monitoring. 

One benefit to using `gym-deepmimic` to visualize is that any policy can be visualized. In `DeepMimic.py`, only policies that were trained by AMP in TensorFlow can be loaded, tested, and visualized. 
#### Enabling rendering
Unlike many gym environments, rendering needs to be enabled by setting the `enable_draw` parameter in the `__init__` constructor to True. The reason for this is that `DeepMimicCore` initializes in a different manner when it needs to handle visualization. When rendering is disabled, GLUT will not be initialized and setup so the window for visualization will not show up. Additionally, calling `render` will not do anything. The choice to make rendering optional is for users who only want to interact with the environment and collect trajectories.
