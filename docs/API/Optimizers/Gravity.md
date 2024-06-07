# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - Gravity

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Gravity.new({initialStepSize: number, movingAverage: number}): OptimizerObject

```

#### Parameters:

* initialStepSize: The value to set the initial velocity during the first iteration.

* movingAverage: The value that controls the smoothing of gradients during training.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### setParameters()

```

Gravity:setParameters({initialStepSize: number, movingAverage: number})

```

#### Parameters:

* initialStepSize: The value to set the initial velocity during the first iteration.

* movingAverage: The value that controls the smoothing of gradients during training.

## Inherited From:

* [BaseOptimizer](BaseOptimizer.md)
