# [API Reference](../../API.md) - [DropoutBlocks](../DropoutBlocks.md) - DropoutND

## Constructors

### new()

Creates a new dropout block object.

```

DropoutND.new({dropoutRate: number}): DropoutBlockObject

```

#### Parameters:

* dropoutRate: The rate at which the input values are converted to zero.

#### Returns:

* DropoutBlock: The generated dropout block object.

## Functions

```

DropoutND:setParameters({dropoutRate: number})

```

#### Parameters:

* dropoutRate: The rate at which the input values are converted to zero.

## Inherited From

* [BaseDropoutBlock](../BaseDropoutBlock.md)
