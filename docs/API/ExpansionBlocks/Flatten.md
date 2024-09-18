# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Flatten

## Constructors

### new()

Creates a new shape transformation block object.

```

Flatten.new({dimensionArray: {number}}): ShapeTransformationBlockObject

```

#### Parameters:

* dimensionArray: An array containing 2 values to flatten the tensor. The first value contains the starting dimension number and the second value contains the ending dimension number.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
