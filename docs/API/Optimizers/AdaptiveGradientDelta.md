# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveGradientDelta (AdaDelta)

## Constructors

### new()

Creates a new optimizer object.

```

AdaptiveGradientDelta.new({decayRate: number, epsilon: number}): OptimizerObject

```

#### Parameters:

* decayRate: The value that controls the rate of decay.

* epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

AdaptiveGradientDelta:setParameters({decayRate: number, epsilon: number})

```

#### Parameters:

* decayRate: The value that controls the rate of decay.

* epsilon: The value to ensure that the numbers are not divided by zero.

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
