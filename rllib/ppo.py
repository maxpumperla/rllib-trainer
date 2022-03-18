from rllib.trainer import TrainerConfig
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ppo import PPOTrainer


class PPOConfig(TrainerConfig):
    """
    Defines a PPOTrainer from the given configuration

    Args:

        use_critic: Should use a critic as a baseline (otherwise don't use value baseline;
                    required for using GAE).
        use_gae: If true, use the Generalized Advantage Estimator (GAE)
                 with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        lambda: The GAE (lambda) parameter.
        kl_coeff: Initial coefficient for KL divergence.
        rollout_fragment_length: Size of batches collected from each worker.
        train_batch_size: Number of timesteps collected for each SGD round. This defines the size
                          of each SGD epoch.

    Example:
        >>> from rllib.ppo import PPOConfig
        >>> config = PPOConfig().training(kl_coeff=0.3, gamma=0.9, lr=0.01)\
                        .resources(num_gpus=0)\
                        .workers(num_workers=4)
        >>> print(config.to_dict())
        >>> trainer = config.build(env="CartPole-v1")
        >>> trainer.train()

    Example:
        >>> from rllib.ppo import PPOConfig
        >>> trainer = PPOConfig().build(env="CartPole-v1")
        >>> config_dict = trainer.get_config()
        >>>
        >>> config_dict.update({
              "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            }),
        >>> tune.run(
                "PPO",
                stop={"episode_reward_mean": 200},
                config=config_dict,
            )
    """

    def __init__(self,
                 use_critic: bool = True,
                 use_gae: bool = True,
                 lambda_: float = 1.0,
                 kl_coeff: float = 0.2,
                 rollout_fragment_length: int = 200,
                 train_batch_size: int = 4000):

        super().__init__()

        self.use_critic = use_critic
        self.use_gae = use_gae
        self.lambda_ = lambda_
        self.kl_coeff = kl_coeff
        self.rollout_fragment_length = rollout_fragment_length
        self.train_batch_size = train_batch_size

    def to_dict(self):

        import copy
        extra_config = copy.deepcopy(vars(self))
        # Worst naming convention ever. NEVER EVER use reserved key-words...
        extra_config["lambda"] = self.lambda_
        extra_config.pop("lambda_")

        base_config = PPOTrainer.get_default_config()

        return Trainer.merge_trainer_configs(
            base_config, extra_config, _allow_unknown_configs=False)

    def build(self, env=None, logger_creator=None):
        """ Builds a Trainer from the TrainerConfig.

        Args:
            env: Name of the environment to use (e.g. a gym-registered str),
                a full class path (e.g.
                "ray.rllib.examples.env.random_env.RandomEnv"), or an Env
                class directly. Note that this arg can also be specified via
                the "env" key in `config`.
            logger_creator: Callable that creates a ray.tune.Logger
                object. If unspecified, a default logger is created.

        Returns:
            A ray.rllib.agents.ppo.PPOTrainer object.
        """
        return PPOTrainer(config=self.to_dict(), env=env, logger_creator=logger_creator)


if __name__ == "__main__":
    import doctest
    doctest.run_docstring_examples(PPOConfig, globals())
