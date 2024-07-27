# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Reshape

## Constructors

### new()

Creates a new shape transformation block object.

```

Reshape.new({dimensionSizeArray: {integer}}): ShapeTransformationBlockObject

```

#### Parameters:

* dimensionSizeArray: The dimension size array to convert a flattened tensor to a shaped tensor.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Functions

### setParameters()

```

Reshape:setParameters({dimensionSizeArray: {integer}})

```

#### Parameters:

* dimensionSizeArray: The dimension size array to convert a flattened tensor to a shaped tensor.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
