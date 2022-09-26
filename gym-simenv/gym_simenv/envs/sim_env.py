import numpy as np
from deepmimic.env.deepmimic_env import DeepMimicEnv
from gym import Env, spaces
import copy
import torch
from gym import utils
from gym.utils import seeding
from deepmimic.util.arg_parser import ArgParser
import json
from enum import Enum


class SimEnv(Env, utils.EzPickle):
    '''
    This class is a gym wrapper around the dynamics ensemble for MILO. Since MILO is an imitation algorithm,.
    the assumption is that imitation is the only objetcive and there is no goal (as defined by AMP).

    The assumption is that the character is humanoid. The behavior is undefined for non-humanoid
    characters.
    '''

    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        TRAIN_END = 2

    def __init__(self, dynamic_ensemble, deepmimic_args=None, enable_velocity_check=False,
                 horizon=300, device=torch.device('cpu'), seed=None,
                 reset_args={'custom_time': False, 'time_min': 0, 'time_max': 0, 'resolve': True,
                             'noise_bef_rot': False, 'noise_min': 0,
                             'noise_max': 0, 'radian': 0, 'rot_vel_w_pose': False, 'vel_noise': False, 'interp': False,
                             'knee_rot': False}):
        '''
        Constructor for SimEnv

        Parameter dynamics_ensemble: This ensemble models the dynamics of the scene specified by deepmimic_args.
        Parameter deepmimic_args: A .txt file specifying the scene to load in DeepMimicEnv. There are many examples given
                                    in the DeepMimic directory. This should be the same argument file used to collect the data
                                    that was used to train the dynamics_ensemble
        Parameter enable_velocity_check: (bool) If enabled, a third condition is enabled for termination of the state.
                                          That is, if any velocity specified by the state exceeds a threshold, terminate the state.
        Parameter horizon: (int) Horizon of the trajectory. Default 300 since in testing, we set the horizon of the spinkick to 10s
                            Since the policy in DeepMimic runs at 30Hz, this corresponds to 300 timesteps.
        Parameter device: Location to load dynamics ensemble.
        Parameter seed: (int) Seed used for seeding numpy and deepmimic env
        Parameter reset_args: (dict) This dictionary contains parameters used for resetting DeepMimicEnv.
        '''
        utils.EzPickle.__init__(**locals())

        #####Load DeepMimicEnv#####
        self.deepmimic = DeepMimicEnv(['--arg_file', deepmimic_args], False)
        self.deepmimic.set_mode(self.Mode.TEST)  # 1 for test
        self.seed_env(seed)

        self.dynamic_ensemble = dynamic_ensemble  # Dynamic Model should be on device
        self.device = device
        self.enable_velocity_check = enable_velocity_check
        self.horizon = horizon

        ######Setup states####
        self.ob = None
        self.num_steps = 0
        self.agentID = 0 #Assumes there is only one agent which is why it is hard-coded to query agent 0

        #####Setup Observation and Action#####
        self.state_size = self.deepmimic.get_state_size(self.agentID)
        self.action_size = self.deepmimic.get_action_size(self.agentID)
        self.observation_space = spaces.Box(low=np.array([-np.inf] * self.state_size),
                                            high=np.array([np.inf] * self.state_size),
                                            dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([-np.inf] * self.action_size),
                                       high=np.array([np.inf] * self.action_size),
                                       dtype=np.float64)

        #####Reset args#####
        self.time_min = reset_args['time_min'] if reset_args['custom_time'] else 0
        self.time_max = reset_args['time_max'] if reset_args['custom_time'] else self.deepmimic.get_motion_length()
        self.reset_dict = dict(time=0, resolve=reset_args['resolve'], noise_bef_rot=reset_args['noise_bef_rot'],
                               low=reset_args['noise_min'], high=reset_args['noise_max'], radian=reset_args['radian'],
                               rot_vel_w_pose=reset_args['rot_vel_w_pose'], vel_noise=reset_args['vel_noise'],
                               interp=reset_args['interp'], knee_rot=reset_args['knee_rot'])

        #####Controller params#####
        self.arg_parser = ArgParser()
        self.arg_parser.load_file(deepmimic_args)
        with open(self.arg_parser.parse_string('char_ctrl_files')) as f:
            ctrl_json = json.load(f)
        self.record_vel_as_pos = bool(ctrl_json.get('RecordVelAsPos', False))
        self.record_all_world = bool(ctrl_json.get('RecordAllWorld', False))
        self.record_world_root_pos = bool(ctrl_json.get('RecordWorldRootPos', False))
        self.record_world_root_rot = bool(ctrl_json.get('RecordWorldRootRot', False))
        self.sampling_rate = 1.0 / self.deepmimic.get_agent_update_rate()

        #####Character Data#####
        with open(self.arg_parser.parse_string('character_files')) as f:
            humanoid_json = json.load(f)
        self.body_defs = humanoid_json['BodyDefs']
        self.pos_dim = self.deepmimic.get_pos_feature_dim()  # Should be 3
        self.rot_dim = self.deepmimic.get_rot_feature_dim()  # Should be 6

        # These are the offsets to the start of certain joints in the position/angle part of the state.
        self.fall_contact_bodies = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14])
        self.fall_contact_bodies_offset = (
                                                      self.pos_dim + self.rot_dim) * self.fall_contact_bodies + 1  # Add 1 since 0th element is root y pos

        self.fall_contact_bodies_params = []
        self.fall_contact_bodies_shapes = []
        for index in self.fall_contact_bodies:
            params = []
            params.append(humanoid_json['BodyDefs'][index]['Param0'])
            params.append(humanoid_json['BodyDefs'][index]['Param1'])
            params.append(humanoid_json['BodyDefs'][index]['Param2'])
            self.fall_contact_bodies_params.append(params)
            self.fall_contact_bodies_shapes.append(humanoid_json['BodyDefs'][index]['Shape'])
        self.fall_contact_bodies_params = self.fall_contact_bodies_params

        #####Set active model in ensemble#####
        self.reset_counter = 0
        self.dynamics = dynamic_ensemble.models[0]
        self.dynamics.model.eval()

    def seed_env(self, seed=None):
        '''
        If seed none, we don't overwrite deepmimic seed. Deepmimic code 
        automatically randomizes seed so if we want to have reproducibility, need to 
        manually seed here. 
        
        Additionally, if seed is none, numpy will be randomly seeded. 
        '''
        if seed:
            self.deepmimic.seed(seed)
        self.np_random, seed = seeding.np_random(seed)

    def get_observation(self):
        return self.ob

    def set_observation(self, value):
        self.ob = value

    def step(self, action):
        '''
        Steps forward using the dynamics ensemble.

        Returns the next state, reward, done flag, and infos.

        The reward returned is always 0 since MILO doesn't actually use the reward signal from this environment.

        To determine termination, we check three conditions in is_done function.

        The infos will be an empty dictionary.
        '''
        assert (self.ob is not None)
        self.num_steps += 1
        with torch.no_grad():
            state_input = torch.from_numpy(self.ob).float().unsqueeze(0).to(self.device)
            action_input = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
            state_diff = self.dynamics.forward(state_input, action_input, unnormalize_out=True)
        self.ob += state_diff.squeeze(0).cpu().numpy()

        reward = 0
        done = self.is_done()
        return copy.deepcopy(self.ob), reward, done, {}

    def is_done(self):
        '''
        To determine termination, we check if we've exceeded the horizon, if any part of the character defined
        by the state has touched the ground (estimate), and if any velocity of the links defined in the state
        exceed in the threshold (if enable_velocity_check is true).
        '''
        horizon_done = self.num_steps >= self.horizon
        collided = self.check_collision()
        velocity_exploded = self.check_velocity() if self.enable_velocity_check else False
        return horizon_done or collided or velocity_exploded

    def check_sphere(self, index):
        '''
        Check if the sphere defined by index has touched the ground.
        '''
        assert self.fall_contact_bodies_shapes[index] == 'sphere'
        offset = self.fall_contact_bodies_offset[index]
        if self.record_all_world or (index == 0 and self.record_world_root_pos):
            body_part_world_y = self.ob[offset + 1]
        else:
            root_world_y = self.ob[0]
            body_part_relative_y = self.ob[offset + 1]
            body_part_world_y = root_world_y + body_part_relative_y

        radius = 0.5 * self.fall_contact_bodies_params[index][0]
        return body_part_world_y <= radius + 0.0001  # 0.001 added as buffer.

    def check_capsule(self, index):
        '''
        Checks if the capsule defined by index has touched the ground by computing
        the two endpoints of the cylinder and
        '''
        assert self.fall_contact_bodies_shapes[index] == 'capsule'
        offset = self.fall_contact_bodies_offset[index]
        # Get center point of the circles ends of the cylinder. Cylinder is still centered at origin here.
        radius = 0.5 * self.fall_contact_bodies_params[index][0]
        cylinder_height = self.fall_contact_bodies_params[index][1]

        '''
        self.ob[offset + selfpos_dim: offset+self+pos_dim+3] contain the "norm" angle. Instead of storing
        the 4D quaternion directly in the state by 2 vectors, the norm and tangent vector. The norm vector
        is simply the vector (0,1,0) rotated by quaternion. 
        
        To get the 2 endpoints of the cylindrical part of the capsule, we first assume the cylinder is 
        centered at the origin. We multiply the the vector (0,1,0) by 0.5*cylinder_height and -0.5cylinder_height
        and rotate by the quaternion. Since the x- and z- coordinate are 0, we can just take the y-coordinate and do this so
        we just do 0.5*cylinder_height*norm_y since norm_y has already been rotated. 
        '''

        norm_y = self.ob[offset + self.pos_dim + 1]
        # Get two endpoints of cylinder (assuming capsule is centered at origin)
        top_cap_center_y = 0.5 * cylinder_height * norm_y
        bottom_cap_center_y = -0.5 * cylinder_height * norm_y

        '''
        We shift these two points by body_part_world_y, which represents the world coordinates of the center, which
        gives us the world coordinates of these 2 endpoints.
        '''

        if self.record_all_world or (index == 0 and self.record_world_root_pos):
            # Get world position of capsule center
            body_part_world_y = self.ob[offset + 1]
        else:
            # Get relative position of capsule center and add root world position to get world position of capsule center.
            root_world_y = self.ob[0]
            body_part_relative_y = self.ob[offset + 1]
            body_part_world_y = body_part_relative_y + root_world_y
        # Get world positions of centers of two endcaps of cylinder.
        body_part_top_cap_center_y = body_part_world_y + top_cap_center_y
        body_part_bottom_cap_center_y = body_part_world_y + bottom_cap_center_y

        # Check if any point in either sphere caps is less than radius from ground
        return body_part_top_cap_center_y <= radius + 0.0001 or body_part_bottom_cap_center_y <= radius + 0.0001

    def check_box(self, index):
        '''
        This does not need to be implemented as the only body parts that are represented by boxes are
        the ankles and we do not care if these touch the ground
        '''
        # TODO: Implement if character is non-humanoid
        return False

    def check_collision(self):
        '''
        Iterates through all possible fall contact points (links) and checks if any of them have collided with the ground.
        '''
        collided = False
        for i in range(len(self.fall_contact_bodies)):
            if self.fall_contact_bodies_shapes[i] == 'sphere':
                collided |= self.check_sphere(i)
            elif self.fall_contact_bodies_shapes[i] == 'capsule':
                collided |= self.check_capsule(i)
            # Add more shapes here if necessary.
        return collided

    def check_velocity(self, threshold=100):
        '''
        Checks velocities of all links in state and checks if they are above threshold (in abslolute value).
        '''
        vel_offset = self.deepmimic.get_vel_offset()
        velocity = self.ob[vel_offset:]
        if self.record_vel_as_pos:
            # If true, the velocity in the state was already multiplied by dt so need to divide to get vel
            velocity /= self.sampling_rate
        return np.any(np.abs(velocity) > threshold)

    def reset(self):
        '''
        Resets the characters by quering DeepMimicCore (DeepMimicEnv python frontend) using the
        parameters defined by the reset_args stored in __init__.
        '''
        # Reset to random time expert pose
        time = self.np_random.uniform(low=0, high=self.time_max)
        self.num_steps = 0
        self.reset_dict['time'] = time
        self.deepmimic.reset_time(**self.reset_dict)
        self.ob = self.deepmimic.record_state(0)
        # update model to use
        self.reset_counter = (self.reset_counter + 1) % len(self.dynamic_ensemble.models)
        self.dynamics = self.dynamic_ensemble.models[self.reset_counter]
        self.dynamics.model.eval()
        return copy.deepcopy(self.ob)

    def render(self, mode='human', close=False):
        pass
