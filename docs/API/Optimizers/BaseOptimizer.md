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

### setCalculateFunction()

Sets a calculate function for the base optimizer.

```

BaseOptimizer:setCalculateFunction(CalculateFunction)

```

#### Parameters:

* The calculate function to be used by the base optimizer when calculate() function is called.

### getOptimizerInternalParameterArray()

Gets the optimizer's internal parameters from the base optimizer.

```
BaseOptimizer:getOptimizerInternalParameterArray(doNotDeepCopy: boolean): {}
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the optimizer internal parameters.

#### Returns:

* optimizerInternalParameterArray: The optimizer internal parameters that is stored in base optimizer.

### setOptimizerInternalParameterArray()

Sets the optimizer's internal parameters from the base optimizer.

```
BaseOptimizer:setOptimizerInternalParameterArray(optimizerInternalParameterArray: {}, doNotDeepCopy: boolean)
```

#### Parameters:

* optimizerInternalParameterArray: The optimizer internal parameters that is stored to be stored in base optimizer.

* doNotDeepCopy: Set whether or not to deep copy the optimizer internal parameters.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
BaseOptimizer:reset()
```

## Inherited From:

* [BaseInstance](../Cores/BaseInstance.md)
