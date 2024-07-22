# [API Reference](../../API.md) - [OperatorBlocks](../OperatorBlocks.md) - ZScoreNormalize

## Constructors

### new()

Creates a new operator block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

ZScoreNormalize.new({dimension: number}): OperatorBlockObject

```

Parameters:

* dimension: The dimension to ZScoreNormalize across the tensor.

#### Returns:

* OperatorBlock: The generated compression block object.

## Functions

### setParameters()

```

ZScoreNormalize:setParameters({dimension: number})

```

#### Parameters:

* dimension: The dimension to ZScoreNormalize across the tensor.

## Inherited From

* [BaseOperatorBlock](BaseOperatorBlock.md)
