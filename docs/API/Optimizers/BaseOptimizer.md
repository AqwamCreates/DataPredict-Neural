# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - BaseOptimizer

BaseOptimizer is a base for all optimizers.

## Constructors

### new()

Creates a new base optimizer object.

```

BaseOptimizer.new(): OptimizerObject

```

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

## calculate()

Returns a modified cost function derivatives.

```

BaseOptimizer:calculate(learningRate: number, costFunctionDerivativeTensor: tensor): tensor

```

#### Parameters:

* learningRate: The learning rate used by a model.

* costFunctionDerivativeTensor: The cost function derivatives calculated by a model.

#### Returns:

* costFunctionDerivativeTensor: The modified cost function derivatives that is to be used by a model.

### reset()

Reset optimizer's stored values (excluding the parameters).

```

BaseOptimizer:reset()

```

### setCalculateFunction()

Sets a calculate function for the base optimizer.

```

BaseOptimizer:setCalculateFunction(CalculateFunction)

```

#### Parameters:

* The calculate function to be used by the base optimizer when calculate() function is called.

### setResetFunction()

Sets a reset function for the base optimizer.

```

BaseOptimizer:setResetFunction(ResetFunction)

```

#### Parameters:

* The reset function to be used by the base optimizer when reset() function is called.

## Inherited From:

* [BaseInstance](../Cores/BaseInstance.md)
