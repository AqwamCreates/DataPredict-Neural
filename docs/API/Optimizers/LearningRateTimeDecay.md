# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - LearningRateTimeDecay

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```

LearningRateTimeDecay.new({decayRate: number, timeStepToDecay: integer}): OptimizerObject

```

#### Property Table Parameters:

* decayRate: The decay rate for learning rate.

* timeStepToDecay: The number of time steps to decay the learning rate.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

LearningRateTimeDecay:setParameters({decayRate: number, timeStepToDecay: integer})

```

#### Parameters:

* decayRate: The decay rate for learning rate.

* timeStepToDecay: The number of time steps to decay the learning rate.

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
