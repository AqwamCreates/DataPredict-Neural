# [API Reference](../../API.md) - [PoolingBlocks](../PoolingBlocks.md) - MaximumPooling3D (MaxPooling3D)

## Constructors

### new()

Creates a new pooling block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

MaximumPooling3D.new({kernelDimensionSizeArray: {integer}, strideDimensionSizeArray: {integer}, outputSizeRoundingMode: string}): PoolingBlockObject

```

#### Parameters:

* kernelDimensionSizeArray: The dimension size for the kernel. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. Note that the stride moves along one axis completely before incrementing to the next axis.

* outputSizeRoundingMode: The rounding mode that determines how the dimension size of the transformed tensor. Available options are:

	* Floor (Default)

	* Ceil

#### Returns:

* PoolingBlock: The generated pooling block object.

## Functions

### setParameters()

```

MaximumPooling3D:setParameters({kernelDimensionSizeArray: {integer}, strideDimensionSizeArray: {integer}, outputSizeRoundingMode: string})

```

#### Parameters:

* kernelDimensionSizeArray: The dimension size for the kernel. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. Note that the stride moves along one axis completely before incrementing to the next axis.

* outputSizeRoundingMode: The rounding mode that determines how the dimension size of the transformed tensor. Available options are:

	* Floor

	* Ceil

## Inherited From

* [BasePoolingBlock](BasePoolingBlock.md)
