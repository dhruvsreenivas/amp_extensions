from gym.envs.registration import register

register(
    id="simenv-v0",
    entry_point="gym_simenv.envs:SimEnv"
    )