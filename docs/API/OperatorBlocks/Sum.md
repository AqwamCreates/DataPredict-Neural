# [API Reference](../../API.md) - [OperatorBlocks](../OperatorBlocks.md) - Sum

## Constructors

### new()

Creates a new compression block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Sum.new({dimension: number}): CompressionBlockObject

```

Parameters:

* dimension: The dimension to sum across the tensor.

#### Returns:

* CompressionBlock: The generated compression block object.

## Functions

### setParameters()

```

Sum:setParameters({dimension: number})

```

#### Parameters:

* dimension: The dimension to sum across the tensor.

## Inherited From

* [BaseOperatorBlock](BaseOperatorBlock.md)