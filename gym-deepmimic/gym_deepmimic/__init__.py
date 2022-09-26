from gym.envs.registration import register

register(
    id="deepmimic-v0",
    entry_point="gym_deepmimic.envs:DeepMimicGymEnv"
    )