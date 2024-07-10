# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - AutoSizeBias (ASBias)

## Constructors

### new()

Creates a new weight block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Bias.new({learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject}): WeightBlockObject

```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

#### Returns

WeightBlock: The generated weight block object.

## Inherited From:

* [BaseWeightBlock](BaseWeightBlock.md)
