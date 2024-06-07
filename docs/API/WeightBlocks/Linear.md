# [API Reference](../../API.md) - WeightBlocks(../WeightBlocks.md) - Linear

## Constructors

### new()

Creates a new weight block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Linear.new({dimensionArray: {number}, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject}): WeightBlockObject

```

#### Parameters:

* dimensionArray: The dimension for the weights. The length of array represents the number of dimensions. The value at the specified index represents the size at that dimension.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

#### Returns:

WeightBlock: The generated weight block object.

## Inherited From:

* [BaseWeightBlock](BaseWeightBlock.md)
