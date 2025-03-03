# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - DataPredictLinearAndBias (DPLAB)

## Constructors

### new()

Creates a new weight block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

DataPredictLinearAndBias.new({dimensionArray: {number}, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject, initializationMode: string}): WeightBlockObject

```

#### Parameters:

* dimensionSizeArray: The dimensions for the weights. The length of array represents the number of dimensions. The value at the specified index represents the size at that dimension.

* hasBiasAtCurrentLayer: Set whether or not there is bias at the current layer.

* hasBiasAtNextLayer: Set whether or not there is bias at the next layer.

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

* WeightBlock: The generated weight block object.

## Inherited From

* [BaseWeightBlock](BaseWeightBlock.md)
