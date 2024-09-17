# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Permute

## Constructors

### new()

Creates a new shape tranformation block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Permute.new({dimensionArray: {number}}): ShapeTransformationBlockObject

```

#### Parameters:

* dimensionArray: A table containing the dimensions to swap sizes with.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Functions

### setParameters()

```

Permute:setParameters({dimensionArray: {number}})

```

#### Parameters:

* dimensionArray: A table containing the dimensions to swap sizes with.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
