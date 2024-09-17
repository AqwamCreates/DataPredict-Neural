# [API Reference](../../API.md) - [Models](../Models.md) - DeepExpectedStateActionRewardStateAction (Deep Expected SARSA)

DeepExpectedStateActionRewardStateAction is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepExpectedStateActionRewardStateAction.new({epsilon: number, discountFactor: number}): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* epsilon: Controls the balance between exploration and exploitation for calculating expected Q-values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepExpectedStateActionRewardStateAction:setParameters({epsilon: number, discountFactor: number})
```

#### Parameters:

* epsilon: Controls the balance between exploration and exploitation for calculating expected Q-values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningBaseModel](ReinforcementLearningBaseModel.md)

## References

* [Expected SARSA in Reinforcement Learning](https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
