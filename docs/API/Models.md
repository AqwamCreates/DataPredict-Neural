# [API Reference](../API.md) - Models

## Deep Reinforcement Learning

| Model                                                                                                                            | Alternate Names                           | Use Cases                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                                         | Deep Q Network                            | Self-Learning Fighting AIs, Self-Learning Parkouring AIs, Self-Driving Cars |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                                         | Double Deep Q Network (2010)              | Same As Deep Q-Learning                                                     |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                                         | Double Deep Q Network (2015)              | Same As Deep Q-Learning                                                     |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                                               | Clipped Double Deep Q Network             | Same As Deep Q-Learning                                                     |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                                   | Deep SARSA                                | Same As Deep Q-Learning                                                     |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                                   | Double Deep SARSA                         | Same As Deep Q-Learning                                                     |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                                   | Double Deep SARSA                         | Same As Deep Q-Learning                                                     |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                                   | Deep Expected SARSA                       | Same As Deep Q-Learning                                                     |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md)                   | Double Deep Expected SARSA                | Same As Deep Q-Learning                                                     |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md)                   | Double Deep Expected SARSA                | Same As Deep Q-Learning                                                     |
| [MonteCarloControl](Models/MonteCarloControl.md)                                                                                 | None                                      | Same As Deep Q-Learning                                                     |
| [OffPolicyMonteCarloControl](Models/OffPolicyMonteCarloControl.md)                                                               | None                                      | Same As Deep Q-Learning                                                     |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                                         | VPG                                       | Same As Deep Q-Learning                                                     |
| [REINFORCE](Models/REINFORCE.md)                                                                                                 | None                                      | Same As Deep Q-Learning                                                     |
| [ActorCritic](Models/ActorCritic.md)                                                                                             | AC                                        | Same As Deep Q-Learning                                                     |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                                           | A2C                                       | Same As Deep Q-Learning                                                     |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                                     | SAC                                       | Same As Deep Q-Learning                                                     |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                                               | PPO                                       | Same As Deep Q-Learning                                                     |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                                       | PPO-Clip                                  | Same As Deep Q-Learning                                                     |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                                     | DDPG                                      | Same As Deep Q-Learning                                                     |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)                               | TD3                                       | Same As Deep Q-Learning                                                     |

## Generative

| Model                                                                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------|
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                                 | GAN             | Building And Art Generation           |
| [ConditionalGenerativeAdversarialNetwork](Models/ConditionalGenerativeAdversarialNetwork.md)                           | CGAN            | Same As GAN, But Can Assign Classes   |
| [WassersteinGenerativeAdversarialNetwork](Models/WassersteinGenerativeAdversarialNetwork.md)                           | WGAN            | Same As GAN, But More Stable          |
| [ConditionalWassersteinGenerativeAdversarialNetwork](Models/ConditionalWassersteinGenerativeAdversarialNetwork.md)     | CWGAN           | Combination Of Both CGAN And WGAN     |

## Others

| Model                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------|-----------------|---------------------------------------|
| [RandomNetworkDistillation](Models/RandomNetworkDistillation.md)       | RND             | Intrinsic Reward Generation           |

## BaseModels

[BaseModel](Models/BaseModel.md)

[ReinforcementLearningBaseModel](Models/ReinforcementLearningBaseModel.md)

[ReinforcementLearningActorCriticBaseModel](Models/ReinforcementLearningActorCriticBaseModel.md)
