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
        """Initializes a PPOConfig instance.
        """
        super().__init__(trainer_class=DQNTrainer)

        if hiddens is None:
            hiddens = [256]

        self.dueling = dueling
        self.hiddens = hiddens
        self.double_q = double_q
        self.n_step = n_step


if __name__ == "__main__":
    import doctest
    doctest.run_docstring_examples(DQNConfig, globals())
