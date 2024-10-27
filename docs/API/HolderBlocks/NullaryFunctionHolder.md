# [API Reference](../../API.md) - [HolderBlocks](../HolderBlocks.md) - NullaryFunctionHolder

## Constructors

### new()

Creates a new holder block object.

```

BaseHolderBlock.new({Function: function, ChainRuleFirstDerivativeFunction: function, FirstDerivativeFunction: function}): HolderBlockObject

```

#### Parameters:

* Function: The nullary function where the function generates an output without using any inputs.

* ChainRuleFirstDerivativeFunction: The chain rule first derivative function of the nullary function.

* FirstDerivativeFunction: The first derivative function of the nullary function.

#### Returns:

* HolderBlock: The generated holder block object.

## Inherited From

* [BaseHolderBlock](BaseHolderBlock.md)
