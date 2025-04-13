# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - LearningRateStepDecay

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```

LearningRateStepDecay.new({decayRate: number, timeStepToDecay: integer}): OptimizerObject

```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate.

* decayRate: The decay rate for learning rate.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

LearningRateStepDecay:setParameters({decayRate: number, timeStepToDecay: integer})

```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate.

* decayRate: The decay rate for learning rate.

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
