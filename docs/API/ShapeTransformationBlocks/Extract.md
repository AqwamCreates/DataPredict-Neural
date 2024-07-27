# [API Reference](../../API.md) - [Reshape](../ShapeTransformationBlocks.md) - Extract

## Constructors

### new()

Creates a new shape transformation block object.

```

Reshape.new({originDimensionIndexArray: {integer}, targetDimensionIndexArray: {integer}}): ShapeTransformationBlockObject

```

#### Parameters:

* originDimensionIndexArray: The origin points to extract the tensor from.

* targetDimensionIndexArray: The target points to extract the tensor from.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Functions

### setParameters()

```

Reshape:setParameters({originDimensionIndexArray: {integer}, targetDimensionIndexArray: {integer}})

```

#### Parameters:

* originDimensionIndexArray: The origin points to extract the tensor from.

* targetDimensionIndexArray: The target points to extract the tensor from.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
