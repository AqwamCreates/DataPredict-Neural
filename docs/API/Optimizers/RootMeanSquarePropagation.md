# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - RootMeanSquarePropagation (RMSProp)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```

RootMeanSquarePropagation.new({beta: number, epsilon: number}): OptimizerObject

```

#### Parameters:

* beta: The value that controls the exponential decay rate for the moving average of squared gradients.

* epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

RootMeanSquarePropagation:setParameters({beta: number, epsilon: number})

```

#### Parameters:

* beta: The value that controls the exponential decay rate for the moving average of squared gradients.

* epsilon: The value to ensure that the numbers are not divided by zero.

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
