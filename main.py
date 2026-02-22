from DQN.dddqn import DDDQN
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
training_period = 250
num_training_episodes = 10000
logging.basicConfig(level=logging.INFO, format='%(message)s')
env = gym.make("CartPole-v1", render_mode="rgb_array")


env = RecordVideo(
    env,
    video_folder="cartpole-training",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0 
)
env = RecordEpisodeStatistics(env)

print(f"Starting training for {num_training_episodes} episodes")
print(f"Videos will be recorded every {training_period} episodes")
print(f"Videos saved to: cartpole-training/")
agent = DDDQN(env,False,10000,128,10000,lr = 1e-4)
agent.train()
env.close()