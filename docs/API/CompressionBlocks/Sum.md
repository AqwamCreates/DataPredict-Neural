# [API Reference](../../API.md) - [CompressionBlocks](../CompressionBlocks.md) - Sum

## Constructors

### new()

Creates a new compression block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Sum.new({dimension: number}): CompressionBlockObject

```

Parameters:

* dimensionIndex: The dimension to sum across the tensor.

#### Returns:

* CompressionBlock: The generated compression block object.

## Functions

### setParameters()

```

Sum:setParameters({dimension: number})

```

#### Parameters:

* dimensionIndex: The dimension to sum across the tensor.

## Inherited From

* [BaseCompressionBlock](BaseCompressionBlock.md)
