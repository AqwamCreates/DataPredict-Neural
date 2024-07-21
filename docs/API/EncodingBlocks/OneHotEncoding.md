# [API Reference](../../API.md) - [EncodingBlocks](../EncodingBlocks.md) - OneHotEncoding

## Constructors

### new()

Creates a new encoding block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

OneHotEncoding.new({dimensionIndex: number}): EncodingBlockObject

```

Parameters:

* dimensionIndex: The dimension to sum across the tensor.

#### Returns:

* EncodingBlock: The generated encoding block object.

## Functions

### setParameters()

```

OneHotEncoding:setParameters({dimensionIndex: number})

```

#### Parameters:

* dimensionIndex: The dimension to sum across the tensor.

## Inherited From

* [BaseCompressionBlock](BaseCompressionBlock.md)
