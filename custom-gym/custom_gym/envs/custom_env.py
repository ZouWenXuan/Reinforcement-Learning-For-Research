import gym

class CustomEnv(gym.Env):
    def __init__(self):
        print('CustomEnv Environment initialized')
    def step(self):
        print('CustomEnv Step successful!')
    def reset(self):
        print('CustomEnv Environment reset')
        
class CustomEnvExtend(gym.Env):
    def __init__(self):
        print('CustomEnvExtend Environment initialized')
    def step(self):
        print('CustomEnvExtend Step successful!')
    def reset(self):
        print('CustomEnvExtend Environment reset')