# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleStateActionRewardStateActionV2 (Double Deep SARSA)

DeepDoubleStateActionRewardStateActionV2 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleStateActionRewardStateAction.new({epsilon: number, averagingRate: number, discountFactor: number}): ModelObject
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDoubleStateActionRewardStateAction:setParameters({epsilon: number, averagingRate: number, discountFactor: number})
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningBaseModel](ReinforcementLearningBaseModel.md)

## References

* [Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning](https://www.scirp.org/journal/paperinformation.aspx?paperid=71237)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
