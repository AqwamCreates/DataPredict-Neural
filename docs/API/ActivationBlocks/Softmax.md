# [API Reference](../../API.md) - [ActivationBlocks](../ActivationBlocks.md) - Softmax

## Constructors

### new()

Creates a new activation block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Softmax.new({dimensionIndex: number}): ActivationBlockObject

```

Parameters:

* dimensionIndex: The dimension to sum across the tensor.

#### Returns:

* ActivationBlock: The generated activation block object.

## Functions

### setParameters()

```

Softmax:setParameters({dimensionIndex: number})

```

#### Parameters:

* dimensionIndex: The dimension to sum across the tensor.

## Inherited From:

* [BaseActivationBlock](BaseActivationBlock.md)
