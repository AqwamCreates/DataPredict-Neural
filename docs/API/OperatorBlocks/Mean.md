# [API Reference](../../API.md) - [OperatorBlocks](../OperatorBlocks.md) - Mean

## Constructors

### new()

Creates a new operator block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Mean.new({dimension: number}): OperatorBlockObject

```

Parameters:

* dimension: The dimension to Mean across the tensor.

#### Returns:

* OperatorBlock: The generated compression block object.

## Functions

### setParameters()

```

Mean:setParameters({dimension: number})

```

#### Parameters:

* dimension: The dimension to Mean across the tensor.

## Inherited From

* [BaseOperatorBlock](BaseOperatorBlock.md)
