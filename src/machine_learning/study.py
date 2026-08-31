import logging
import time

import gymnasium as gym

logger = logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    logger.info("obs = %s", obs)
    logger.info("action = %s", action)
    logger.info("reward = %s", reward)

    total_reward += reward
    # if input() == "q":
    #     break
    done = terminated or truncated
    time.sleep(0.5)

logger.info("Total reward: %s ", total_reward)

env.close()
