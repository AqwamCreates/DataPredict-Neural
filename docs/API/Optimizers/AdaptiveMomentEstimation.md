# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveMomentEstimation (Adam)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```

AdaptiveMomentEstimation.new({beta1: number, beta2: number, epsilon: number}): OptimizerObject

```

#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients.

* beta2: The decay rate of the moving average of the squared gradients.

* epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

AdaptiveMomentEstimation:setParameters({beta1: number, beta2: number, epsilon: number})

```
#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients.

* beta2: The decay rate of the moving average of the squared gradients.

* epsilon: The value to ensure that the numbers are not divided by zero.

## Inherited From:

* [BaseOptimizer](BaseOptimizer.md)
