# [API Reference](../../API.md) - [ConvolutionBlock](../ConvolutionBlock.md) - Convolution1D

## Constructors

### new()

Creates a new convolution block object.

```

Convolution1D.new({numberOfKernels: number, kernelDimensionSize: number, strideDimensionSize: number, outputSizeRoundingMode: number}): ConvolutionBlockObject

```

#### Parameters:

* numberOfKernels: The number of kernels to be used to extract the features from input tensor.

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* outputSizeRoundingMode: The rounding mode that determines how the dimension size of the transformed tensor. Available options are:

	* Floor (Default)

	* Ceil

#### Returns:

* ConvolutionBlock: The generated convolution block object.

## Inherited From

* [BaseConvolutionBlock](../BaseConvolutionBlock.md)
