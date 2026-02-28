from POLICY_OPTIM.ppo import PPO
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo,ClipAction
env = gym.make("CartPole-v1")
env = ClipAction(env)
agent = PPO(env,False,True,200,lr = 1e-4)
agent.train()
env.close()