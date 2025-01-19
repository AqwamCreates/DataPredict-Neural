# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - AutomaticBias (AutoBias)

## Constructors

### new()

Creates a new weight block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

AutomaticBias.new({numberOfDimensions: number, shareBiasDimensionArray: {number}, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject, initializationMode: string}): WeightBlockObject

```

#### Parameters:

* numberOfDimensions: The number of dimensions in which the bias values will be applied to. The bias values will be applied on the last dimension first.

* shareBiasDimensionArray: An array containing dimensions where the bias are to be shared.

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

#### Returns

* WeightBlock: The generated weight block object.

## Inherited From

* [BaseWeightBlock](BaseWeightBlock.md)
