# [API Reference](../../API.md) - [ConvolutionBlocks](../ConvolutionBlocks.md) - Convolution3D

## Constructors

### new()

Creates a new convolution block object.

```

Convolution3D.new({numberOfKernels: number, kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}, outputSizeRoundingMode: {number}}): ConvolutionBlockObject

```

#### Parameters:

* numberOfKernels: The number of kernels to be used to extract the features from input tensor.

* kernelDimensionSizeArray: The dimension size for the kernel. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. Note that the stride moves along one axis completely before incrementing to the next axis.

* outputSizeRoundingMode: The rounding mode that determines how the dimension size of the transformed tensor. Available options are:

	* Floor (Default)

	* Ceil

#### Returns:

* ConvolutionBlock: The generated convolution block object.

## Inherited From

* [BaseConvolutionBlock](../BaseConvolutionBlock.md)
