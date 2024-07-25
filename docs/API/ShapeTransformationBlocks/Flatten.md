# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Flatten

## Constructors

### new()

Creates a new shape transformation block object.

```

Flatten.new({startDimension: number, endDimension: number}): ShapeTransformationBlockObject

```

#### Parameters:

* startDimension: The starting dimension to flatten into a single tensor.

* endDimension: The ending dimension to flatten into a single tensor.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
