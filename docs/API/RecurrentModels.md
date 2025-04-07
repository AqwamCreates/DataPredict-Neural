# [API Reference](../API.md) - Models

## Recurrent Deep Reinforcement Learning

* Note that all of these recurrent models require RecurrentNeuralNetworkCell or GatedRecurrentUnitCell containers. It is recommended to use the former since it uses less computational resources than the latter.

* Currently, these recurrent models have no documentation. Fortunately, you can still refer to the non-recurrent versions of these models.

* Additionally, they cannot work with DataPredict's QuickSetups for deep reinforcement learning. You'll have to use the classic setup to use the recurrent models.

| Model                                                                                                                                                     | Alternate Names                           | Use Cases                                                                   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [RecurrentDeepQLearning](RecurrentModels/RecurrentDeepQLearning.md)                                                                                         | Recurrent Deep Q Network                | Self-Learning Fighting AIs, Self-Learning Parkouring AIs, Self-Driving Cars |
| [RecurrentDeepDoubleQLearningV1](RecurrentModels/RecurrentDeepDoubleQLearningV1.md)                                                                         | Recurrent Double Deep Q Network (2010)  | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepDoubleQLearningV2](RecurrentModels/RecurrentDeepDoubleQLearningV2.md)                                                                         | Recurrent Double Deep Q Network (2015)  | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepClippedDoubleQLearning](RecurrentModels/RecurrentDeepClippedDoubleQLearning.md)                                                               | Recurrent Clipped Double Deep Q Network | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepStateActionRewardStateAction](RecurrentModels/RecurrentDeepStateActionRewardStateAction.md)                                                   | Recurrent Deep SARSA                    | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepDoubleStateActionRewardStateActionV1](RecurrentModels/RecurrentDeepDoubleStateActionRewardStateActionV1.md)                                   | Recurrent Double Deep SARSA             | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepDoubleStateActionRewardStateActionV2](RecurrentModels/RecurrentDeepDoubleStateActionRewardStateActionV2.md)                                   | Recurrent Double Deep SARSA             | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepExpectedStateActionRewardStateAction](RecurrentModels/RecurrentDeepExpectedStateActionRewardStateAction.md)                                   | Recurrent Deep Expected SARSA           | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepDoubleExpectedStateActionRewardStateActionV1](RecurrentModels/RecurrentDeepDoubleExpectedStateActionRewardStateActionV1.md)                   | Recurrent Double Deep Expected SARSA    | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepDoubleExpectedStateActionRewardStateActionV2](RecurrentModels/RecurrentDeepDoubleExpectedStateActionRewardStateActionV2.md)                   | Recurrent Double Deep Expected SARSA    | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentMonteCarloControl](RecurrentModels/RecurrentMonteCarloControl.md)                                                                                 | None                                    | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentOffPolicyMonteCarloControl](RecurrentModels/RecurrentOffPolicyMonteCarloControl.md)                                                               | None                                    | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentVanillaPolicyGradient](RecurrentModels/RecurrentVanillaPolicyGradient.md)                                                                         | Recurrent VPG                           | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentREINFORCE](RecurrentModels/RecurrentREINFORCE.md)                                                                                                 | None                                    | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentActorCritic](RecurrentModels/RecurrentActorCritic.md)                                                                                             | Recurrent AC                            | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentAdvantageActorCritic](RecurrentModels/RecurrentAdvantageActorCritic.md)                                                                           | RecurrentA2C                            | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentSoftActorCritic](RecurrentModels/RecurrentSoftActorCritic.md)                                                                                     | Recurrent SAC                           | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentProximalPolicyOptimization](RecurrentModels/RecurrentProximalPolicyOptimization.md)                                                               | Recurrent PPO                           | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentProximalPolicyOptimizationClip](RecurrentModels/RecurrentProximalPolicyOptimizationClip.md)                                                       | RecurrentPPO-Clip                       | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentDeepDeterministicPolicyGradient](RecurrentModels/RecurrentDeepDeterministicPolicyGradient.md)                                                     | Recurrent DDPG                          | Same As Recurrent Deep Q-Learning                                           |
| [RecurrentTwinDelayedDeepDeterministicPolicyGradient](RecurrentModels/RecurrentTwinDelayedDeepDeterministicPolicyGradient.md)                               | Recurrent TD3                           | Same As Recurrent Deep Q-Learning                                           |

## BaseModels

[RecurrentReinforcementLearningBaseModel](RecurrentModels/RecurrentReinforcementLearningBaseModel.md)

[RecurrentReinforcementLearningActorCriticBaseModel](RecurrentModels/RecurrentReinforcementLearningActorCriticBaseModel.md)

[DualRecurrentReinforcementLearningBaseModel](RecurrentModels/DualRecurrentReinforcementLearningBaseModel.md)

[DualRecurrentReinforcementLearningActorCriticBaseModel](RecurrentModels/DualRecurrentReinforcementLearningActorCriticBaseModel.md)
