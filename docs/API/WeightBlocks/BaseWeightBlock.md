# [API Reference](../../API.md) - [WeightBlocks](../WeightBlocks.md) - Bias

## Constructors

### new()

```

BaseWeightBlock.new(): WeightBlockObject

```

#### Returns

WeightBlock: The generated weight block object.

## Functions

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

* tensor: Tensor to be returned.

### setNextFunctionBlockArrayIndexArray()

```

BaseWeightBlock:setNextFunctionBlockArrayIndexArray(nextFunctionBlockArrayIndexArray: {number})

```

#### Parameters:

* nextFunctionBlockArrayIndexArray: The array that determines the path to the next function block. The first index is the starting path.

### getNextFunctionBlockArrayIndexArray()

```

BaseWeightBlock:getNextFunctionBlockArrayIndexArray()

```

#### Returns:

* nextFunctionBlockArrayIndexArray: The array that determines the path to the next function block. The first index is the starting path.

### setWaitDuration()

```

BaseWeightBlock:setNextFunctionBlockWaitDuration(waitDuration: number)

```

#### Parameters:

* nextFunctionBlockWaitDuration: The duration to wait for a tensor from next function blocks before timeout.

### getWaitDuration()

```

BaseWeightBlock:getNextFunctionBlockWaitDuration(): number

```

#### Returns:

* nextFunctionBlockWaitDuration: The duration to wait for a tensor from next function blocks before timeout.

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

* Regularizer: The regularizer to be used.

### initializeLayer()

```

BaseWeightBlock:initializeLayer({dimensionSizeArray: {number}, learningRate: number, Optimizer: OptimizerObject, Regularizer: RegularizerObject, nextFunctionBlockArrayIndexArray: {number}, nextFunctionBlockWaitDuration: number})

```

#### Parameters:

* dimensionSizeArray: The dimensions for the weights. The length of array represents the number of dimensions. The value at the specified index represents the size at that dimension.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer to be used.

* Regularizer: The regularizer to be used.

* nextFunctionBlockArrayIndexArray: The array that determines the path to the next function block. The first index is the starting path.

* nextFunctionBlockWaitDuration: The duration to wait for a tensor from next function blocks before timeout.

### waitForTransformedTensorRecursive()

```lua

BaseWeightBlock:waitForTransformedTensorRecursive(CurrentFunctionBlock: FunctionBlock, nextFunctionBlockArrayIndexArray: {number})

```

#### Parameters:

* CurrentFunctionBlock: The current function block that is being accessed.

* nextFunctionBlockArrayIndexArray: The array that determines the path to the next function block. The first index is the starting path.

## Inherited From:

* [FunctionBlock](../Cores/FunctionBlock.md)
