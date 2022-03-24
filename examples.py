import ray
from ray import tune

from rllib.dqn import DQNConfig
from rllib.ppo import PPOConfig

#ray.init(local_mode=True)

# Manual RLlib Trainer setup.
dqn_config = DQNConfig() \
    .training(gamma=0.9, lr=0.01) \
    .resources(num_gpus=0) \
    .workers(num_workers=1)
dqn_trainer = dqn_config.build(env="CartPole-v1")
print(dqn_trainer.train())

# With tune.
ppo_config = PPOConfig(kl_coeff=0.1).environment(env="CartPole-v1")
ppo_config.training(lr=tune.grid_search([0.001, 0.0001]))

tune.run(
    "PPO",
    stop={"episode_reward_mean": 150.0},
    config=ppo_config.to_dict()
)
