# =============================================================================
# Q1: Example of DQN
# =============================================================================

#%% Environment
import gym
env_name = "LunarLander-v2"

env = gym.make(env_name)

episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()          
    done = False
    score = 0

    while not done:
        env.render(mode='human')             
        action = env.action_space.sample()    
        n_state, reward, done, info = env.step(action)    
        score += reward
    print("Episode : {}, Score : {}".format(episode, score))

env.close()



#%% RL: stable baselines3
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy 

# device test
print(stable_baselines3.common.utils.get_device(device='auto'))

# set env
env_name = "LunarLander-v2"
env = gym.make(env_name)
env = DummyVecEnv([lambda : env])
env = VecNormalize(env, norm_obs=False, norm_reward=False, training=True)

# set model
model = DQN(
    "MlpPolicy", 
    env=env, 
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    gamma = 0.99,
    target_update_interval=250,
    train_freq = 4,
    gradient_steps = -1,
    exploration_fraction = 0.12,
    exploration_final_eps = 0.1,
    policy_kwargs={"net_arch" : [256, 256]},
    verbose=2,
    tensorboard_log="./tensorboard/LunarLander-v2/"
)

model.learn(total_timesteps=1e5)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
model.save("LunarLander3.pkl")
env.close()


#%% Save and load
env = gym.make(env_name)
model = DQN.load("LunarLander3.pkl")

state = env.reset()
done = False 
score = 0
while not done:
    action, _ = model.predict(observation=state)
    state, reward, done, info = env.step(action=action)
    score += reward
    env.render()
env.close()

