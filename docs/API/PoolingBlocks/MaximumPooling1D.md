# [API Reference](../../API.md) - [PoolingBlocks](../PoolingBlocks.md) - MaximumPooling1D (MaxPooling1D)

## Constructors

### new()

Creates a new pooling block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

MaximumPooling1D.new({kernelDimensionSize: integer, strideDimensionSize: integer}): PoolingBlockObject

```

#### Parameters:

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. 

#### Returns:

* PoolingBlock: The generated pooling block object.

## Functions

### setParameters()

```

MaximumPooling1D:setParameters({kernelDimensionSize: integer, strideDimensionSize: integer, outputSizeRoundingMode: string})

```

#### Parameters:

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* outputSizeRoundingMode: The rounding mode that determines how the dimension size of the transformed tensor. Available options are:

	* Floor

	* Ceil

## Inherited From

* [BasePoolingBlock](BasePoolingBlock.md)