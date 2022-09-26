import gym
from gym import spaces
from gym import Env
from deepmimic.env.deepmimic_env import DeepMimicEnv
import numpy as np
from enum import Enum
from gym import utils
from gym.utils import seeding
import copy
from deepmimic.util.arg_parser import ArgParser


win_width = 800
win_height = int(win_width * 9.0 / 16.0)
reshaping = False
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class DeepMimicGymEnv(Env, utils.EzPickle):
    """
    Custom Environment that follows gym interface for the AMP/Deepmimic framework. This has only been tested
    with the AMP and DeepMimic imitation-only scenes. However, the environment should work fine as is or only
    need minor adjustments to work with scenes that contain additional goal tasks
    """
    metadata = {'render.modes': ['human']}

    # Copied from RLAgent in DeepMimic/learning. Needed to set deepmimic env mode.
    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        TRAIN_END = 2

    def __init__(self, deepmimic_args=None, enable_draw=False,
                 seed=None,reset_args={'custom_time': False, 'time_min': 0, 'time_max': 0, 'resolve': True,
                             'noise_bef_rot': False, 'noise_min': 0,
                             'noise_max': 0, 'radian': 0, 'rot_vel_w_pose': False, 'vel_noise': False, 'interp': False,
                             'knee_rot': False}):
        '''
        Constructor for DeepMimicGymEnv
        Parameter deepmimic_args: A .txt file specifying the scene to load in DeepMimicEnv. There are many examples given
                                    in the DeepMimic directory. This should be the same argument file used to collect the data
                                    that was used to train the dynamics_ensemble
        Parameter enable_draw: (bool) If draw, initializes GLUT and also sets up drawing in DeepMimicCore. Render function will
        display current state. If false, GLUT not initalized, no window will show, and calling render does nothing.
        Parameter seed: (int) Seed used for seeding numpy and deepmimic env
        Parameter reset_args: (dict) This dictionary contains parameters used for resetting DeepMimicEnv.
        '''
        utils.EzPickle.__init__(**locals())

        #Initialize GLUT if rendering enabled.
        self.enable_draw = enable_draw
        if enable_draw:
            self._init_draw()

        # Load DeepMimicEnv
        self.deepmimic = DeepMimicEnv(['--arg_file', deepmimic_args], enable_draw)
        self.deepmimic.set_mode(self.Mode.TEST)  # 1 for test
        self.seed_env(seed)

        #Setup GLUT. This needs to be after DeepMimicEnv/DeepMimicCore is initialized.
        if enable_draw:
            self._setup_draw()

        self.arg_parser = ArgParser()
        self.arg_parser.load_file(deepmimic_args)
        self.imitate_amp = self.arg_parser.parse_string('scene')=='imitate_amp'
        self.agentID = 0 #Assumes there is only one agent which is why it is hard-coded to query agent 0


        # Set state/action space
        self.state_size = self.deepmimic.get_state_size(self.agentID)
        self.action_size = self.deepmimic.get_action_size(self.agentID)
        self.goal_size = self.deepmimic.get_goal_size(self.agentID)
        self.observation_space = spaces.Box(low=np.array([-np.inf] * self.state_size), high=np.array([np.inf] * self.state_size),
                                            dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([-np.inf] * self.action_size), high=np.array([np.inf] * self.action_size),
                                       dtype=np.float64)
        self.goal_space= spaces.Box(low=np.array([-np.inf] * self.goal_size), high=np.array([np.inf] * self.goal_size),
                                       dtype=np.float64)


        # Set variables used for updating
        '''
        The frequency of sampling sholud be 30 Hz. 
        
        The paper says the world simulation updates at a frequency of 1.2kHz. The time we pass to 
        deepmimic.update should be 1/600 if we follow what DeepMimic.py. To get this, 
        the timestep for each update should be 1/60. This timestep of 1/60 is divided into 10 substeps which gives us a 
        timestep update to the DeepMimicCore of 1/600. Then, the arg file specifies that the Bullet
        further takes this time and does 2 simulation substeps so the Bullet world actually updates with time
        1/1200 or 1.2 kHz. 
        '''
        self.sampling_rate = 1.0/self.deepmimic.get_agent_update_rate()
        assert self.deepmimic.get_agent_update_rate()==30
        '''
        self.update_timestep is hardcoded to get the world update frequency of 1.2kHz. This comes from DeepMimic.py 
        and is originally the frequency at which DeepMimic.update_world is called. Our step function can be thought of as doing 2 DeepMimic.update_world
        calls since the timestep between policy queries is 1/30
        '''
        self.update_timestep = 1.0/60 #This is hardcoded to get the world update freqency to be 1.2 kHz. This comes from DeepMimic.py
        self.num_substeps = self.deepmimic.get_num_update_substeps() #Should be 10

        self.total_substeps = int((self.sampling_rate / self.update_timestep) * self.num_substeps)
        self.substep_timestep = self.update_timestep / self.num_substeps

        self.ob = None

        #Set reset args
        self.time_min = reset_args['time_min'] if reset_args['custom_time'] else 0
        self.time_max = reset_args['time_max'] if reset_args['custom_time'] else self.deepmimic.get_motion_length()
        self.reset_dict = dict(time=0, resolve=reset_args['resolve'], noise_bef_rot=reset_args['noise_bef_rot'],
                               low=reset_args['noise_min'], high=reset_args['noise_max'], radian=reset_args['radian'],
                               rot_vel_w_pose=reset_args['rot_vel_w_pose'], vel_noise=reset_args['vel_noise'],
                               interp=reset_args['interp'], knee_rot=reset_args['knee_rot'])


    def seed_env(self, seed=None):
        '''
        Seed numpy used for resetting env and deepmimic
        '''
        if seed:
            self.deepmimic.seed(seed)
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        '''
        The step function is based on update_world in DeepMimic.py. Assuming that update_timestep=1/60, num_substeps=10,
        and sampling_rate=30 which are the defaults in DeepMimic. This step function corresponds to two calls
        of update_world(world, 1/60) as our agent operates at a frequency of 30Hz.
        '''
        assert(self.ob is not None)
        # TODO: Add goals and also add DTW cost into info. Check goals
        self.deepmimic.set_action(self.agentID, action)
        done = False
        info = {'valid': True}
        for _ in range(self.total_substeps):
            self.deepmimic.update(self.substep_timestep)
            valid_episode = self.deepmimic.check_valid_episode()
            if valid_episode:
                end_episode = self.deepmimic.is_episode_end()
                if end_episode:
                    done = True
                    break
            else:
                done = True
                info['valid'] = False
                break

        state = self.deepmimic.record_state(self.agentID)
        if self.goal_size > 0:
            state = np.concatenate((state, self.deepmimic.record_goal(self.agentID)))
        reward = self.deepmimic.calc_reward(self.agentID)
        self.ob = state
        if self.imitate_amp and done:
            #In this case, reward will be dtw cost so set it to 0 and save DTW cost in info dictionary.
            info['dtw_cost'] = reward
            reward = 0
        #info['amp_obs'] = self.deepmimic.record_amp_obs_agent(0) uncomment to AMP discriminator feature (consists of current and last state features concatanted in zig-zag fashion).
        #info['amo_obs_current'] = self.deepmimic.record_amp_obs_agent_current(0) uncomment to get AMP discriminator feature of only current state
        return copy.deepcopy(state), reward, done, info



    def reset(self):
        '''
        If seed none, we don't overwrite deepmimic seed. Deepmimic code 
        automatically randomizes seed so if we want to have reproducibility, need to 
        manually seed here. 
        
        Additionally, if seed is none, numpy will be randomly seeded. 
        '''
        time = self.np_random.uniform(low=self.time_min, high=self.time_max)
        self.reset_dict['time'] = time
        self.deepmimic.reset_time(**self.reset_dict)
        self.ob = self.deepmimic.record_state(self.agentID)
        return copy.deepcopy(self.ob)

    def _init_draw(self):
        glutInit()

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(win_width, win_height)
        glutCreateWindow(b'DeepMimic')

        return

    def _update_intermediate_buffer(self):
        global gym_env
        if not (reshaping):
            if (win_width != self.deepmimic.get_win_width() or win_height != self.deepmimic.get_win_height()):
                self.deepmimic.reshape(win_width, win_height)

        return

    def _draw(self):
        global reshaping
        global gym_env
        self._update_intermediate_buffer()
        self.deepmimic.draw()

        glutSwapBuffers()
        reshaping = False

        return

    def reshape(self, w, h):
        global reshaping
        global win_width
        global win_height

        reshaping = True
        win_width = w
        win_height = h

        return

    def _setup_draw(self):
        glutDisplayFunc(self._draw)
        glutReshapeFunc(self._reshape)

        self._reshape(win_width, win_height)
        self.deepmimic.reshape(win_width, win_height)

        return

    def render(self):
        if self.enable_draw:
            glutPostRedisplay()
            glutMainLoopEvent()