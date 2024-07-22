# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Transpose

## Constructors

### new()

Creates a new shape tranformation block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Transpose.new({dimensionArray: {number}}): ShapeTransformationBlockObject

```

#### Parameters:

* dimensionArray: A table containing a pair of dimensions to swap sizes with. Must contain two values only.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Functions

### setParameters()

```

Transpose:setParameters({dimensionArray: {number}})

```

#### Parameters:

* dimensionArray: A table containing a pair of dimensions to swap sizes with. Must contain two values only.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
