# [API Reference](../../API.md) - [PoolingBlocks](../PoolingBlocks.md) - MaximumUnpooling1D (MaxUnpooling1D)

## Constructors

### new()

Creates a new pooling block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

MaximumUnpooling1D.new({kernelDimensionSize: integer, strideDimensionSize: integer, unpoolingMethod: string}): PoolingBlockObject

```

#### Parameters:

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. 

* unpoolingMethod: The unpooling method that determines how the transformed tensor is generated. Available options are:

	* NearestNeighbour (Default)

	* BedOfNails

#### Returns:

* PoolingBlock: The generated pooling block object.

## Functions

### setParameters()

```

MaximumUnpooling1D:setParameters({kernelDimensionSize: integer, strideDimensionSize: integer, unpoolingMethod: string})

```

#### Parameters:

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. 

* unpoolingMethod: The unpooling method that determines how the transformed tensor is generated. Available options are:

	* NearestNeighbour

	* BedOfNails

## Inherited From

* [BasePoolingBlock](BasePoolingBlock.md)

## References

* [Unsampling: Unpooling and Transpose Convolution](https://medium.com/jun94-devpblog/dl-12-unsampling-unpooling-and-transpose-convolution-831dc53687ce)
