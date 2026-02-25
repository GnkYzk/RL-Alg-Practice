from POLICY_OPTIM.actor_critic import ActorCritic
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
env = gym.make("CartPole-v1")
agent = ActorCritic(env,False,True,2000,lr = 1e-4)
agent.train()
env.close()