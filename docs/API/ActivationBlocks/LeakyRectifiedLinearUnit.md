# [API Reference](../../API.md) - ActivationBlocks(../ActivationBlocks.md) - LeakyRectifiedLinearUnit (LeakyReLU)

## Constructors

### new()

Creates a new activation block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

LeakyRectifiedLinearUnit.new({negativeSlopeFactor: number}): ActivationBlockObject

```

Parameters:

* negativeSlopeFactor: The value to be multiplied with negative input values. 

#### Returns:

ActivationBlock: The generated activation block object.

## Functions

### setParameters()

```

LeakyRectifiedLinearUnit:setParameters({negativeSlopeFactor: number})

```

#### Parameters:

* negativeSlopeFactor: The value to be multiplied with negative input values. 

## Inherited From:

* [BaseActivationBlock](BaseActivationBlock.md)
