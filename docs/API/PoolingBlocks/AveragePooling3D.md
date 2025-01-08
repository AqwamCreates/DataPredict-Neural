# [API Reference](../../API.md) - [PoolingBlocks](../PoolingBlocks.md) - AveragePooling3D (AvgPooling3D)

## Constructors

### new()

Creates a new pooling block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

AveragePooling3D.new({kernelDimensionSizeArray: {integer}, strideDimensionSizeArray: {integer}}): PoolingBlockObject

```

#### Parameters:

* kernelDimensionSizeArray: The dimension size for the kernel. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. Note that the stride moves along one axis completely before incrementing to the next axis.

#### Returns:

* PoolingBlock: The generated pooling block object.

## Inherited From

* [BasePoolingBlock](BasePoolingBlock.md)
