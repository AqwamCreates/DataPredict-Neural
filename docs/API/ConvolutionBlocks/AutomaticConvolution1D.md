# [API Reference](../../API.md) - [ConvolutionBlocks](../ConvolutionBlocks.md) - AutomaticConvolution1D (AutoConvolution1D)

## Constructors

### new()

Creates a new convolution block object.

```

AutomaticConvolution1D.new({numberOfKernels: number kernelDimensionSize: number, strideDimensionSize: number, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject, initializationMode: string}): ConvolutionBlockObject

```

#### Parameters:

* numberOfKernels: The number of kernels to be used to extract the features from the input tensor.

* kernelDimensionSize: The dimension size for the kernel.

* strideDimensionSize: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. 

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

* initializationMode: The mode for the weights to be initialized. Available options are:

	* Zero

	* Random

	* RandomNormal

	* RandomUniformPositive

	* RandomUniformNegative

	* RandomUniformNegativeAndPositive

	* HeUniform

	* HeNormal

	* XavierUniform

	* XavierNormal

	* LeCunUniform

	* LeCunNormal
	
	* None

#### Returns:

* ConvolutionBlock: The generated convolution block object.

## Inherited From

* [BaseConvolutionBlock](BaseConvolutionBlock.md)
