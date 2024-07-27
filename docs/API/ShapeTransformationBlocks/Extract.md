# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Extract

## Constructors

### new()

Creates a new shape transformation block object.

```

Reshape.new({originDimensionIndexArray: {integer}, targetDimensionIndexArray: {integer}}): ShapeTransformationBlockObject

```

#### Parameters:

* originDimensionIndexArray: The origin points to extract the tensor from. It must have the same number of dimensions as the input tensor.

* targetDimensionIndexArray: The target points to extract the tensor from. It must have the same number of dimensions as the input tensor.

#### Returns:

* ShapeTransformationBlock: The generated shape transformation block object.

## Functions

### setParameters()

```

Reshape:setParameters({originDimensionIndexArray: {integer}, targetDimensionIndexArray: {integer}})

```

#### Parameters:

* originDimensionIndexArray: The origin points to extract the tensor from. It must have the same number of dimensions as the input tensor.

* targetDimensionIndexArray: The target points to extract the tensor from. It must have the same number of dimensions as the input tensor.

## Inherited From

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
