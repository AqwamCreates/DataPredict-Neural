# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - AutomaticLinear (AutoLinear)

## Constructors

### new()

Creates a new weight block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

AutomaticLinear.new({finalDimensionSize: number, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject}): WeightBlockObject

```

#### Parameters:

* finalDimensionSize: The dimension for final dimension of the weight tensor. 

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

#### Returns:

* WeightBlock: The generated weight block object.

## Inherited From

* [BaseWeightBlock](BaseWeightBlock.md)
