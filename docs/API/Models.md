# [API Reference](../API.md) - Models

## Deep Reinforcement Learning

| Model                                                                                                                            | Alternate Names                           | Use Cases                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                                         | Deep Q Network                            | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                                         | Double Deep Q Network (2010)              | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                                         | Double Deep Q Network (2015)              | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                                               | Clipped Double Deep Q Network             | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                                   | Deep SARSA                                | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                                   | Double Deep SARSA                         | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                                   | Double Deep SARSA                         | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                                   | Deep Expected SARSA                       | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md)                   | Double Deep Expected SARSA                | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md)                   | Double Deep Expected SARSA                | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [ActorCritic](Models/ActorCritic.md)                                                                                             | AC                                        | Critic-Based Self-Learning Player AIs                                       |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                                           | A2C                                       | Advantage-Based Self-Learning Player AIs                                    |
| [REINFORCE](Models/REINFORCE.md)                                                                                                 | None                                      | Reward-Based Self-Learning Player AIs                                       |
| [MonteCarloControl](Models/MonteCarloControl.md) (May Need Further Refinement)                                                   | None                                      | Online Self-Learning Player AIs                                             |
| [OffPolicyMonteCarloControl](Models/OffPolicyMonteCarloControl.md)                                                               | None                                      | Offline Self-Learning Player AIs                                            |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                                         | VPG                                       | Baseline-Based Self-Learning Player AIs                                     |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                                               | PPO                                       | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs      |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                                       | PPO-Clip                                  | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs      |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                                     | SAC                                       | Self-Learning Vehicle AIs                                                   |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                                     | DDPG                                      | Self-Learning Vehicle AIs                                                   |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)                               | TD3                                       | Self-Learning Vehicle AIs                                                   |

## Generative

| Model                                                                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------|
| [Diffusion](Models/Diffusion.md)                                                                                       |                 | Building And Image Generation         |
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                                 | GAN             | Building And Image Generation         |
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
