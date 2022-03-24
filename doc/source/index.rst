RLlib Configuration
===================

Link to GitHub project:  `maxpumperla/rllib-trainer <https://github.com/maxpumperla/rllib-trainer>`_

.. contents::
    :local:
    :depth: 1

Defining a trainer
------------------

Here's how you define and run a PPO Trainer, with and without Tune:

.. literalinclude:: /../../examples.py
   :language: python
   :start-after: __trainer_begin__
   :end-before: __trainer_end__


And here's an example for a DQN trainer:

.. literalinclude:: /../../examples.py
   :language: python
   :start-after: __dqn_begin__
   :end-before: __dqn_end__


If you define a DQN Trainer with the wrong config, your IDE will tell you on definition:

.. literalinclude:: /../../examples.py
   :language: python
   :start-after: __dqn_fail_begin__
   :end-before: __dqn_fail_end__

Here's a snapshot from PyCharm:

.. image:: assets/dqn_fail.png
    :align: center

How to document Trainers in a less annoying way
-----------------------------------------------

Instead of having loooooooong lists of flat parameters, we can simply auto-generate
class documentation, with types and stuff.
Users might actually understand what's going on!


TrainerConfigurator
-------------------

.. autoclass:: rllib.trainer.TrainerConfig
    :members:
    :show-inheritance:

PPO
---

.. autoclass:: rllib.ppo.PPOConfig
    :members:
    :show-inheritance:

DQN
---

.. autoclass:: rllib.dqn.DQNConfig
    :members:
    :show-inheritance:
