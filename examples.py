


from rllib.dqn import DQNConfig


config = DQNConfig(kl_coeff=0.3) \
    .training(gamma=0.9, lr=0.01) \
    .resources(num_gpus=0) \
    .workers(num_workers=4)


from ray import tune
from rllib.ppo import PPOConfig


trainer = PPOConfig().build(env="CartPole-v1")
config_dict = trainer.get_config()

config_dict.update({
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
})

tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config=config_dict
)
