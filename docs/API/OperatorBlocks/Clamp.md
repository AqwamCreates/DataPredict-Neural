# [API Reference](../../API.md) - [OperatorBlocks](../OperatorBlocks.md) - Clamp

## Constructors

### new()

Creates a new operator block object.

```

Clamp.new({lowerBoundTensor: number/tensor, upperBoundTensor: number/Tensor}): OperatorBlockObject

```

#### Parameters:

* lowerBoundTensor: The lower bound tensor for the input tensor to compare it to.

* upperBoundTensor: The upper bound tensor for the input tensor to compare it to.

#### Returns:

* OperatorBlock: The generated operator block object.

## Inherited From

* [BaseOperatorBlock](BaseOperatorBlock.md)
