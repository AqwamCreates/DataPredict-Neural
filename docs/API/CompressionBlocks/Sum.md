# [API Reference](../../API.md) - [CompressionBlocks](../CompressionBlocks.md) - Sum

## Constructors

### new()

Creates a new compression block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Sum.new({dimensionIndex: number}): CompressionBlockObject

```

Parameters:

* dimensionIndex: The dimension to sum across the tensor.

#### Returns:

* ActivationBlock: The generated compression block object.

## Functions

### setParameters()

```

Sum:setParameters({dimensionIndex: number})

```

#### Parameters:

* dimensionIndex: The dimension to sum across the tensor.

## Inherited From:

* [BaseCompressionBlock](BaseCompressionBlock.md)
