from rllib.trainer import TrainerConfig
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.dqn.dqn import DQNTrainer


class DQNConfig(TrainerConfig):
    """
    Defines a DQNTrainer from the given configuration

    Args:

        dueling (bool): Whether to use dueling architecture
        hiddens: Dense-layer setup for each the advantage branch and the value branch
                in a dueling architecture.
        double_q (bool): Whether to use double Q-learning
        n_step (int): N-step Q learning

    Example:
        >>> from rllib.dqn import DQNConfig
        >>> config = DQNConfig(dueling=False).training(gamma=0.9, lr=0.01)
                        .environment(env="CartPole-v1")
                        .resources(num_gpus=0)
                        .workers(num_workers=4)
        >>> trainer = config.build()
    """
    def __init__(self,
                 dueling=True,
                 hiddens=None,
                 double_q=True,
                 n_step=1,
                 ):

        super().__init__()

        if hiddens is None:
            hiddens = [256]

        self.dueling = dueling
        self.hiddens = hiddens
        self.double_q = double_q
        self.n_step = n_step

    def to_dict(self):

        extra_config = vars(self)
        # Worst naming convention ever. NEVER EVER use reserved key-words...
        extra_config["lambda"] = self.lambda_
        extra_config.pop("lambda_")

        base_config = DQNTrainer.get_default_config()

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
            A ray.rllib.agents.dqn.DQNTrainer object.
        """
        return DQNTrainer(config=self.to_dict(), env=env, logger_creator=logger_creator)


if __name__ == "__main__":
    import doctest
    doctest.run_docstring_examples(PPOConfig, globals())
