# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - Momentum

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Momentum.new({decayRate: number}): OptimizerObject

```
#### Parameters:

* decayRate: The value that controls the rate of decay.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

Momentum:setParameters({decayRate: number})

```

#### Parameters:

* decayRate: The value that controls the rate of decay.

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
