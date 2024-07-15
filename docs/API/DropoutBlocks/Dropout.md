# [API Reference](../../API.md) - [DropoutBlocks](../DropoutBlocks.md) - Dropout

## Constructors

### new()

Creates a new dropout block object.

```

Dropout.new({dropoutRate: number}): DropoutBlockObject

```

#### Parameters:

* dropoutRate: The rate at which the input values are converted to zero.

#### Returns:

* DropoutBlock: The generated dropout block object.

## Functions

```

Dropout:setParameters({dropoutRate: number})

```

#### Parameters:

* dropoutRate: The rate at which the input values are converted to zero.

## Inherited From

* [BaseDropoutBlock](../BaseDropoutBlock.md)
