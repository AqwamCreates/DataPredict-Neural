# [API Reference](../../API.md) - [PoolingBlocks](../PoolingBlocks.md) - MinimumPooling1D (MinPooling1D)

## Constructors

### new()

Creates a new pooling block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

MinimumPooling1D.new({kernelDimensionSize: integer, strideDimensionSize: integer}): PoolingBlockObject

```

#### Parameters:

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. 

#### Returns:

* PoolingBlock: The generated pooling block object.

## Inherited From

* [BasePoolingBlock](BasePoolingBlock.md)
