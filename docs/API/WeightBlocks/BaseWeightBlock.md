# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - Bias

## Constructors

### new()

```

BaseWeightBlock.new({learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject, nextFunctionBlockArrayIndexArray: {number}, nextFunctionBlockWaitDuration: number, initializationMode: string, updateWeightTensorInPlace: boolean}): WeightBlockObject

```

#### Parameters:

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

* updateWeightTensorInPlace: Set whether or not to update the weight tensor in place. If true, updates the weight tensor directly for better performance by avoiding new table creation and reducing memory usage. Not supported for scalar values. [Default: true]

#### Returns

* WeightBlock: The generated weight block object.

## Functions

### gradientDescent()

```

BaseWeightBlock:gradientDescent(weightLossTensor: tensor, numberOfData: number)

```

#### Parameters:

* weightLossTensor: A tensor containing the weight loss values. This will be used to update the weights.

## Functions

### gradientAscent()

```

BaseWeightBlock:gradientAscent(weightLossTensor: tensor, numberOfData: number)

```

#### Parameters:

* weightLossTensor: A tensor containing the weight loss values. This will be used to update the weights.

### setWeightTensor()

```

BaseWeightBlock:setWeightTensor(weightTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters

* weightTensor: The tensor to be loaded to the weight block.

* doNotDeepCopy: Whether or not to deep copy the weight tensor.

### getWeightTensor()

```

BaseWeightBlock:getWeightTensor(doNotDeepCopy): tensor

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the weight tensor.

#### Returns

* weightTensor: Tensor to be returned.

### setLearningRate()

```

BaseWeightBlock:setLearningRate(learningRate: number)

```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

### getLearningRate(): number

```

BaseWeightBlock:getLearningRate()

```

#### Returns::

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

### setOptimizer()

```

BaseWeightBlock:setOptimizer(Optimizer: OptimizerObject)

```

#### Parameters:

* Optimizer: The optimizer to be used.

### getOptimizer()

```

BaseWeightBlock:getOptimizer(): OptimizerObject

```

#### Returns:

* Optimizer: The optimizer used by the weight block.

### setRegularizer()

```

BaseWeightBlock:setRegularizer(Regularizer: RegularizerObject)

```

#### Parameters:

* Regularizer: The regularizer to be used.

### getRegularizer()

```

BaseWeightBlock:getRegularizer(): RegularizerObject

```

#### Returns:

* Regularizer: The regularizer used by the weight block.

## Inherited From

* [BaseFunctionBlock](../Cores/BaseFunctionBlock.md)
