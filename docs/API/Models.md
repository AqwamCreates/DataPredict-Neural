# [API Reference](../API.md) - Models

| Model Type                                                        | Description                                     | Count |
|-------------------------------------------------------------------|-------------------------------------------------|-------|
| [Deep Reinforcement Learning](#deep-reinforcement-learning)       | State-Action Optimization Using Neural Networks | 26    |
| [Generative](#generative)                                         | Feature To Novel Value                          | 4     |
| Total                                                             |                                                 | 30    |

### Legend

| Icon | Name                        | Description                                            |
|------|-----------------------------|--------------------------------------------------------|
| ❗   | Implementation Issue       | The model may have some implementation problems.        |
| 🔰   | Beginner Algorithm         | Commonly taught to beginners.                           |
| 💾   | Data Efficient             | Require few data to train the model.                    |
| ⚡   | Computationally Efficient  | Require few computational resources to train the model. |
| 🛡️   | Noise Resistant            | Can handle randomness / unclean data.                   |
| 🟢   | Online                     | Can adapt real-time.                                    |
| 🟡   | Session-Adaptive / Offline | Can be retrained each session.                          |
| ⚠️   | Assumption-Heavy           | Have restrictive rules on using the model.              |
| ⚙️   | Configuration-Heavy        | Requires a lot of manual configuration to use.          |

## Deep Reinforcement Learning

## Deep Reinforcement Learning

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                                          | Alternate Names               | Properties  | Use Cases                                                                 |
|----------------------------------------------------------------------------------------------------------------|-------------------------------|-------------|---------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                       | Deep Q Network                | 💾 🟢      | Best Self-Learning Player AIs, Best Recommendation Systems                |
| [DeepNStepQLearning](Models/DeepNStepQLearning.md)                                                             | Deep N-Step Q Network          | 💾 🟢      | Best Self-Learning Player AIs, Best Recommendation Systems                |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                       | Double Deep Q Network (2010)  | 💾 🛡️ 🟢   | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                       | Double Deep Q Network (2015)  | 💾 🛡️ 🟢   | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                             | Clipped Deep Double Q Network | 💾 🛡️ 🟢   | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                 | Deep SARSA                    | 🟢          | Safe Self-Learning Player AIs, Safe Recommendation Systems                |
| [DeepNStepStateActionRewardStateAction](Models/DeepNStepStateActionRewardStateAction.md)                       | Deep N-Step SARSA             | 🟢          | Safe Self-Learning Player AIs, Safe Recommendation Systems                |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                 | Double Deep SARSA             | 🛡️ 🟢      | Stable Safe Self-Learning Player AIs, Safe Recommendation Systems         |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                 | Double Deep SARSA             | 🛡️ 🟢      | Stable Safe Self-Learning Player AIs, Safe Recommendation Systems         |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                 | Deep Expected SARSA           | 🟢         | Balanced Self-Learning Player AIs, Balanced Recommendation Systems        |
| [DeepNStepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)            | Deep N-Step Expected SARSA    | 🟢         | Balanced Self-Learning Player AIs, Balanced Recommendation Systems        |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md) | Double Deep Expected SARSA    | 🛡️ 🟢      | Stable Balanced Self-Learning Player AIs, Balanced Recommendation Systems |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md) | Double Deep Expected SARSA    | 🛡️ 🟢      | Stable Balanced Self-Learning Player AIs, Balanced Recommendation Systems |
| [MonteCarloControl](Models/MonteCarloControl.md)                                                               | None                          | ❗ 🟢      | Online Self-Learning Player AIs                                           |
| [OffPolicyMonteCarloControl](Models/OffPolicyMonteCarloControl.md)                                             | None                          | 🟢         | Offline Self-Learning Player AIs                                          |
| [DeepTemporalDifference](Models/DeepTemporalDifference.md)                                                     | TD                            | 🟢         | Priority Systems                                                          |
| [REINFORCE](Models/REINFORCE.md)                                                                               | None                          | 🟢         | Reward-Based Self-Learning Player AIs                          |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                       | VPG                           | ❗ 🟢      | Baseline-Based Self-Learning Player AIs                                   |
| [ActorCritic](Models/ActorCritic.md)                                                                           | AC                            | 🟢         | Critic-Based Self-Learning Player AIs                                     |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                         | A2C                           | 🟢         | Advantage-Based Self-Learning Player AIs                                  |
| [TemporalDifferenceActorCritic](Models/TemporalDifferenceActorCritic.md)                                       | TD-AC                         | 🟢         | Bootsrapped Online Self-Learning Player AIs                               |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                             | PPO                           | 🟢         | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs    |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                     | PPO-Clip                      | 🟢         | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs    |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                   | SAC                           | 💾 🛡️ 🟢  | Self-Learning Vehicle AIs                                                 |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                   | DDPG                          | 🟢         | Self-Learning Vehicle AIs                                                 |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)             | TD3                           | 🟢 🛡️      | Self-Learning Vehicle AIs                                                 |

## Generative

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                                              | Alternate Names | Properties | Use Cases                                 |
|--------------------------------------------------------------------------------------------------------------------|-----------------|------------| ------------------------------------------|
| [Diffusion](Models/Diffusion.md)                                                                                   | None            | 🟢 🟡     | Building And Image Generation             |
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                             | GAN             | 🟢 🟡     | Enemy Data Generation                     |
| [ConditionalGenerativeAdversarialNetwork](Models/ConditionalGenerativeAdversarialNetwork.md)                       | CGAN            | 🟢 🟡     | Conditional Enemy Data Generation         |
| [WassersteinGenerativeAdversarialNetwork](Models/WassersteinGenerativeAdversarialNetwork.md)                       | WGAN            | 🟢 🟡     | Stable Enemy Data Generation              |
| [ConditionalWassersteinGenerativeAdversarialNetwork](Models/ConditionalWassersteinGenerativeAdversarialNetwork.md) | CWGAN           | 🟢 🟡     | Stable Conditional Enemy Data Generation  |

## Others

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                  | Alternate Names |  Properties | Use Cases                             |
|------------------------------------------------------------------------|-----------------|-------------|---------------------------------------|
| [RandomNetworkDistillation](Models/RandomNetworkDistillation.md)       | RND             | 🟢 🟡      | Intrinsic Reward Generation           |

## BaseModels

[BaseModel](Models/BaseModel.md)

[ReinforcementLearningBaseModel](Models/ReinforcementLearningBaseModel.md)

[ReinforcementLearningActorCriticBaseModel](Models/ReinforcementLearningActorCriticBaseModel.md)
