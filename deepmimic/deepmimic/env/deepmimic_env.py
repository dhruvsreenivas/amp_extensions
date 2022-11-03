import numpy as np
from deepmimic.env.env import Env
from deepmimic.DeepMimicCore import DeepMimicCore
from deepmimic.env.action_space import ActionSpace

class DeepMimicEnv(Env):
    def __init__(self, args, enable_draw):
        super().__init__(args, enable_draw)

        self._core = DeepMimicCore.cDeepMimicCore(enable_draw)

        rand_seed = np.random.randint(np.iinfo(np.int32).max)
        self._core.SeedRand(rand_seed)

        self._core.ParseArgs(args)
        self._core.Init()
        return

    def seed(self, s):
        self._core.SeedRand(s)

    def update(self, timestep):
        self._core.Update(timestep)

    def update_sync_timestep(self, timestep):
        self._core.UpdateSyncTimestep(timestep)

    def reset(self):
        self._core.Reset()

    # Added by Albert
    def reset_time(self, time, resolve = True, noise_bef_rot = False, low = 0, high = 0, radian = 0,
		    rot_vel_w_pose = False, vel_noise = False, interp = 1, knee_rot = False):
        '''
        Resets the character to a specific time in the reference mocap clip and then adds noise.

        Parameter time: (float)  time to reset to.
        Parameter resolve: (boolean) If true, DeepMimicCore resolves any ground character intersections by shifting the character up
                                in the y-axis (see SceneSimChar.cpp). This should be left True as this si the original AMP behavior.
        Parameter noise_bef_rot: (boolean) There are two ways to add noise: add a noisy vector to both the internal pose and velocity
                                            state of the character and sample a random rotation to add to each joint.
                                            If this is true, the noisy vector is added before adding the random angles to each joint. Otherwise,
                                            the noisy vector is added after.
        Parameter low, high: (floats) Each element in the noisy vector is sampled from the uniform distribution U(l, h).
                              A noisy vector is added to the pose as well as the velocity states.
        Parameter radian: (float) When adding noise by sampling a random rotation to add to each joint, the angles are sampled from
                          the uniform distribution U(-radian, radian).
        Parameter rot_vel_w_pose: (bool)If true, this will add the same noisy rotation to the angular velocity of the joint that is
                                   added to the joint's rotation.
        Parameter vel_noise: (bool) If true, we also sample random rotations to add to the angular velocities.
        Parameter interp: (float, precondition: should be in [0,1]). If 1, the velocitgy is reset to the same as the
                          the expert which is what AMP does. If 0, the velocity is set to 0. Otherwise, the velocity
                          is set by interpolating between 0 and the expert based on the interp value.
        Parameter knee_rot: (boolean) If true, noise will also be added to the knee joints. This is default false
                            since adding noise to this joint can lead the character to instantly fall over.
        '''
        self._core.ResetTime(time, resolve, noise_bef_rot, low, high, radian, rot_vel_w_pose, vel_noise, interp, knee_rot)

    # Added by Albert
    def reset_index(self, index, resolve=True, noise_bef_rot=False, low=0, high=0, radian=0,
                        rot_vel_w_pose=False, vel_noise=False, interp=1, knee_rot=False):
        '''
        Resets the character to a specific time in the reference mocap clip and then adds noise.

        Parameter time: (float)  time to reset to.
        Parameter resolve: (boolean) If true, DeepMimicCore resolves any ground character intersections by shifting the character up
                                in the y-axis (see SceneSimChar.cpp). This should be left True as this si the original AMP behavior.
        Parameter noise_bef_rot: (boolean) There are two ways to add noise: add a noisy vector to both the internal pose and velocity
                                            state of the character and sample a random rotation to add to each joint.
                                            If this is true, the noisy vector is added before adding the random angles to each joint. Otherwise,
                                            the noisy vector is added after.
        Parameter low, high: (floats) Each element in the noisy vector is sampled from the uniform distribution U(l, h).
                              A noisy vector is added to the pose as well as the velocity states.
        Parameter radian: (float) When adding noise by sampling a random rotation to add to each joint, the angles are sampled from
                          the uniform distribution U(-radian, radian).
        Parameter rot_vel_w_pose: (bool)If true, this will add the same noisy rotation to the angular velocity of the joint that is
                                   added to the joint's rotation.
        Parameter vel_noise: (bool) If true, we also sample random rotations to add to the angular velocities.
        Parameter interp: (float, precondition: should be in [0,1]). If 1, the velocitgy is reset to the same as the
                          the expert which is what AMP does. If 0, the velocity is set to 0. Otherwise, the velocity
                          is set by interpolating between 0 and the expert based on the interp value.
        Parameter knee_rot: (boolean) If true, noise will also be added to the knee joints. This is default false
                            since adding noise to this joint can lead the character to instantly fall over.
        '''
        self._core.ResetIndex(index, resolve, noise_bef_rot, low, high, radian, rot_vel_w_pose, vel_noise, interp, knee_rot)

    def get_time(self):
        return self._core.GetTime()

    def get_name(self):
        return self._core.GetName()

    # Added by Albert
    def get_motion_length(self):
        return self._core.GetMotionLength()

    # Added by Albert
    def get_agent_update_rate(self, agent_id=0):
        return self._core.GetAgentUpdateRate(agent_id)

    # Added by Albert
    def get_pose_offset(self, agent_id=0):
        return self._core.GetStatePoseOffset(agent_id)

    # Added by Albert
    def get_pose_size(self, agent_id=0):
        return self._core.GetStatePoseSize(agent_id)

    # Added by Albert
    def get_vel_offset(self, agent_id=0):
        return self._core.GetStateVelOffset(agent_id)

    # Added by Albert
    def get_vel_size(self, agent_id=0):
        return self._core.GetStateVelSize(agent_id)

    # Added by Albert
    def get_phase_offset(self, agent_id=0):
        return self._core.GetStatePhaseOffset(agent_id)

    # Added by Albert
    def get_phase_size(self, agent_id=0):
        return self._core.GetStatePhaseSize(agent_id)

    # Added by Albert
    def get_pos_feature_dim(self, agent_id=0):
        return self._core.GetPosFeatureDim(agent_id)

    # Added by Albert
    def get_rot_feature_dim(self, agent_id=0):
        return self._core.GetRotFeatureDim(agent_id)

    # rendering and UI interface
    def draw(self):
        self._core.Draw()

    def keyboard(self, key, x, y):
        self._core.Keyboard(key, x, y)

    def mouse_click(self, button, state, x, y):
        self._core.MouseClick(button, state, x, y)

    def mouse_move(self, x, y):
        self._core.MouseMove(x, y)

    def reshape(self, w, h):
        self._core.Reshape(w, h)

    def shutdown(self):
        self._core.Shutdown()

    def is_done(self):
        return self._core.IsDone()

    def set_playback_speed(self, speed):
        self._core.SetPlaybackSpeed(speed)

    def set_updates_per_sec(self, updates_per_sec):
        self._core.SetUpdatesPerSec(updates_per_sec)

    def get_win_width(self):
        return self._core.GetWinWidth()

    def get_win_height(self):
        return self._core.GetWinHeight()

    def get_num_update_substeps(self):
        return self._core.GetNumUpdateSubsteps()

    # rl interface
    def is_rl_scene(self):
        return self._core.IsRLScene()

    def get_num_agents(self):
        return self._core.GetNumAgents()

    def need_new_action(self, agent_id):
        return self._core.NeedNewAction(agent_id)

    def record_state(self, agent_id):
        return np.array(self._core.RecordState(agent_id))

    def record_goal(self, agent_id):
        return np.array(self._core.RecordGoal(agent_id))

    def get_action_space(self, agent_id):
        return ActionSpace(self._core.GetActionSpace(agent_id))
    
    def set_action(self, agent_id, action):
        return self._core.SetAction(agent_id, action.tolist())
    
    def get_state_size(self, agent_id):
        return self._core.GetStateSize(agent_id)

    def get_goal_size(self, agent_id):
        return self._core.GetGoalSize(agent_id)

    def get_action_size(self, agent_id):
        return self._core.GetActionSize(agent_id)

    def get_num_actions(self, agent_id):
        return self._core.GetNumActions(agent_id)

    def build_state_offset(self, agent_id):
        return np.array(self._core.BuildStateOffset(agent_id))

    def build_state_scale(self, agent_id):
        return np.array(self._core.BuildStateScale(agent_id))
    
    def build_goal_offset(self, agent_id):
        return np.array(self._core.BuildGoalOffset(agent_id))

    def build_goal_scale(self, agent_id):
        return np.array(self._core.BuildGoalScale(agent_id))
    
    def build_action_offset(self, agent_id):
        return np.array(self._core.BuildActionOffset(agent_id))

    def build_action_scale(self, agent_id):
        return np.array(self._core.BuildActionScale(agent_id))

    def build_action_bound_min(self, agent_id):
        return np.array(self._core.BuildActionBoundMin(agent_id))

    def build_action_bound_max(self, agent_id):
        return np.array(self._core.BuildActionBoundMax(agent_id))

    def build_state_norm_groups(self, agent_id):
        return np.array(self._core.BuildStateNormGroups(agent_id))

    def build_goal_norm_groups(self, agent_id):
        return np.array(self._core.BuildGoalNormGroups(agent_id))

    def calc_reward(self, agent_id):
        return self._core.CalcReward(agent_id)

    def get_reward_min(self, agent_id):
        return self._core.GetRewardMin(agent_id)

    def get_reward_max(self, agent_id):
        return self._core.GetRewardMax(agent_id)

    def get_reward_fail(self, agent_id):
        return self._core.GetRewardFail(agent_id)

    def get_reward_succ(self, agent_id):
        return self._core.GetRewardSucc(agent_id)
    
    def enable_amp_task_reward(self):
        return self._core.EnableAMPTaskReward()


    # Added by Albert
    def get_amp_obs_pose_size(self):
        return self._core.GetAMPObsPoseSize()

    # Added by Albert
    def get_amp_obs_vel_size(self):
        return self._core.GetAMPObsVelSize()

    def get_amp_obs_size(self):
        return self._core.GetAMPObsSize()

    def get_amp_obs_offset(self):
        return np.array(self._core.GetAMPObsOffset())
    
    def get_amp_obs_scale(self):
        return np.array(self._core.GetAMPObsScale())
    
    def get_amp_obs_norm_group(self):
        return np.array(self._core.GetAMPObsNormGroup())
    
    def record_amp_obs_expert(self, agent_id):
        return np.array(self._core.RecordAMPObsExpert(agent_id))

    # Added by Albert
    def record_amp_obs_agent_current(self, agent_id):
        '''
        Returns the features for the current state (pose and velocity) that the discriminator uses.
        The AMP framework uses record_amp_obs_agent which concatenates the previous and current state features into one vector.
        '''
        return np.array(self._core.RecordAMPObsAgentCurrent(agent_id))

    def record_amp_obs_agent(self, agent_id):
        return np.array(self._core.RecordAMPObsAgent(agent_id))

    # Added by Albert
    def get_dtw_backtrack_path(self, agent_id=0):
        '''
        Returns the cost for each state by doing backtracking on the DP algorithm.
        '''
        return np.array(self._core.GetDTWBacktrackPath(agent_id))

    def is_episode_end(self):
        return self._core.IsEpisodeEnd()

    def check_terminate(self, agent_id):
       return Env.Terminate(self._core.CheckTerminate(agent_id))

    def check_valid_episode(self):
        return self._core.CheckValidEpisode()

    def log_val(self, agent_id, val):
        self._core.LogVal(agent_id, float(val))
        return

    def set_sample_count(self, count):
        self._core.SetSampleCount(count)
        return

    def set_mode(self, mode):
        self._core.SetMode(mode.value)
        return