# [API Reference](../../API.md) - [HolderBlocks](../HolderBlocks.md) - NullaryFunctionHolder

## Constructors

### new()

Creates a new holder block object.

```

BaseHolderBlock.new({Function: function, FirstDerivativeFunction: function, SecondStepFirstDerivativeFunction: function}): HolderBlockObject

```

#### Parameters:

* Function: The nullary function where the function generates an output without using any inputs.

* FirstDerivativeFunction: The first derivative of the nullary function.

* SecondStepFirstDerivativeFunction: The second step of first derivative of the nullary function.

#### Returns:

* HolderBlock: The generated holder block object.

## Inherited From

* [BaseHolderBlock](BaseHolderBlock.md)
