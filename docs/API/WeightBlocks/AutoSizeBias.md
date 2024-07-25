# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - AutoSizeBias (ASBias)

## Constructors

### new()

Creates a new weight block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Bias.new({numberOfDimensions: number, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject}): WeightBlockObject

```

#### Parameters:

* numberOfDimensions: The number of dimensions in which the bias values will be applied to. The bias values will be applied on the last dimension first.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

#### Returns

* WeightBlock: The generated weight block object.

## Inherited From

* [BaseWeightBlock](BaseWeightBlock.md)
