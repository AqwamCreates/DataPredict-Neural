# [API Reference](../../API.md) - [PoolingBlocks](../PoolingBlocks.md) - MaximumUnpooling3D (MaxUnpooling3D)

## Constructors

### new()

Creates a new pooling block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

MaximumUnpooling3D.new({kernelDimensionSizeArray: {integer}, strideDimensionSizeArray: {integer}, unpoolingMethod: string}): PoolingBlockObject

```

#### Parameters:

* kernelDimensionSizeArray: The dimension size for the kernel. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. Note that the stride moves along one axis completely before incrementing to the next axis.

* unpoolingMethod: The unpooling method that determines how the transformed tensor is generated. Available options are:

	* NearestNeighbour (Default)

	* BedOfNails

#### Returns:

* PoolingBlock: The generated pooling block object.

## Inherited From

* [BasePoolingBlock](BasePoolingBlock.md)

## References

* [Unsampling: Unpooling and Transpose Convolution](https://medium.com/jun94-devpblog/dl-12-unsampling-unpooling-and-transpose-convolution-831dc53687ce)
