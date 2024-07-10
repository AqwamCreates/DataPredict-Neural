# [API Reference](../../API.md) - [ShapeTransformationBlocks](../ShapeTransformationBlocks.md) - Transpose

## Constructors

### new()

Creates a new shape tranformation block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

Transpose.new({dimensionIndexArray: {number}}): ShapeTransformationBlockObject

```

#### Parameters:

* dimensionIndexArray: A table containing a pair of dimension indices to swap sizes with. Must contain two values only.

#### Returns:

ShapeTransformationBlock: The generated shape tranformation block object.

## Functions

### setParameters()

```

Transpose:setParameters({dimensionIndexArray: {number}})

```

#### Parameters:

* dimensionIndexArray: A table containing a pair of dimension indices to swap sizes with. Must contain two values only.

## Inherited From:

* [BaseShapeTransformationBlock](BaseShapeTransformationBlock.md)
