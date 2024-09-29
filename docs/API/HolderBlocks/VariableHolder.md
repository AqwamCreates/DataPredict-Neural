# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - Bias

## Constructors

### new()

```

VariableHolder.new({variableTensor: tensor, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject}): HolderBlockObject

```

#### Parameters:

* variableTensor: The tensor to be stored inside the variable holder block.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

#### Returns

* HolderBlock: The generated holder block object.

## Functions

### setParameters()

```

VariableHolder:setParameters({variableTensor: tensor, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject})

```

#### Parameters:

* variableTensor: The tensor to be stored inside the variable holder block.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

### gradientDescent()

```

VariableHolder:gradientDescent(lossTensor: tensor, numberOfData: number)

```

#### Parameters:

* lossTensor: A tensor containing the loss values. This will be used to update the variable values.

* numberOfData: The value to divide with the loss tensors.

### setVariableTensor()

```

VariableHolder:setVariableTensor(variableTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters

* variableTensor: The tensor to be loaded to the weight block.

* doNotDeepCopy: Whether or not to deep copy the weight tensor.

### getVariableTensor()

```

VariableHolder:getVariableTensor(doNotDeepCopy): tensor

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the weight tensor.

#### Returns

* variableTensor: Tensor to be returned.

### setLearningRate()

```

VariableHolder:setLearningRate(learningRate: number)

```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

### getLearningRate(): number

```

VariableHolder:getLearningRate()

```

#### Returns::

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

### setOptimizer()

```

VariableHolder:setOptimizer(Optimizer: OptimizerObject)

```

#### Parameters:

* Optimizer: The optimizer to be used.

### getOptimizer()

```

VariableHolder:getOptimizer(): OptimizerObject

```

#### Returns:

* Optimizer: The optimizer used by the weight block.

### setRegularizer()

```

VariableHolder:setRegularizer(Regularizer: RegularizerObject)

```

#### Parameters:

* Regularizer: The regularizer to be used.

### getRegularizer()

```

VariableHolder:getRegularizer(): RegularizerObject

```

#### Returns:

* Regularizer: The regularizer to be used.

## Inherited From

* [BaseHolderBlock](BaseHolderBlock.md)
