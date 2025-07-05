# [API Reference](../../API.md) - [Containers](../Containers.md) - BaseContainer

## Constructors

### new()

Creates a new container object.

```

BaseContainer.new({ClassesList: {any}, cutOffValue: number, timeDependent: boolean, parallelGradientDescent: boolean, WeightBlockArray: {WeightBlock}, OutputBlockArray: {FunctionBlock}}): ContainerObject

```

#### Parameters:

* timeDependent: An indicator that the container uses datasets that are time dependent. Enabling this will cause the container to always use slower backward propagation method. [Default: False]

* parallelGradientDescent: An indicator that container will perform parallel gradient descent. See fastUpdate() function for more details. [Default: True]

* WeightBlockArray: An array containing all the weight blocks that will be loaded to the container. [Default: {}]

* OutputBlockArray: An array containing all the function blocks that will be loaded to the container. [Default: {}]

#### Returns:

* Container: The generated container object.

## Functions

### forwardPropagate()

```

Sequential:forwardPropagate(featureTensor: tensor): tensor

```

#### Parameters:

* featureTensor: The feature tensor to be used as an input.

#### Returns:

* generatedLabelTensor: The generated label tensor produced from passing across the function blocks inside the container object.

### calculateWeightLossTensorArray()

```

Sequential:backwardPropagate(lossTensor: tensor): {tensor}

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

#### Returns:

* weightLossTensorArray: An array containing the weight loss tensors. The first tensor in the array represents the weight loss tensor for the first weight block.

### gradientDescent()

```

Sequential:gradientDescent(weightLossTensorArray: {tensor}, numberOfData: number)

```

#### Parameters:

* weightLossTensorArray: An array containing the weight loss tensors. The first tensor in the array represents the weight loss tensor for the first weight block.

* numberOfData: The value to divide with the weight loss tensors.

### fastUpdate()

Performs the gradient descent after immediately receiving the weight loss tensor for that particular weight block.

```

Sequential:fastUpdate(lossTensor: tensor)

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

### slowUpdate()

Performs the gradient descent after all weight loss tensors are received from all weight blocks.

```

Sequential:slowUpdate(lossTensor: tensor)

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

### update()

A wrapper function that switches between fastUpdate() and slowUpdate().

```

Sequential:update(lossTensor: tensor)

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

### setWeightTensorArray()

```

Sequential:setWeightTensorArray(weightTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters:

* weightTensorArray: An array containing all the weight tensors to be set to individual weight blocks. The first tensor in the array represents the weight tensor for the first weight block.

* doNotDeepCopy: Set whether or not to deep copy the weight tensors in the weight array.

### getWeightTensorArray()

```

Sequential:getWeightTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the weight tensors from the weight blocks.

### setMultipleInputBlocks()

```

Sequential:setMultipleInputBlocks(...: FunctionBlockObject)

```

Parameters:

* TrainableBlock: The weight blocks to be added to the graph container. Order matters.

### setMultipleInputBlocks()

```

Sequential:setMultipleInputBlocks(...: FunctionBlockObject)

```

Parameters:

* InputBlock: The function blocks to be added to the graph container. Order matters.

### setMultipleOutputBlocks()

```

Sequential:setMultipleOutputBlocks(...: FunctionBlockObject)

```

### Parameters:

* Function: The function blocks to be added to the graph container. Order matters.

### getWeightBlockByIndex()

```

ComputationalGraph:getWeightBlockByIndex(index): FunctionBlockObject

```

#### Parameters:

* index: The index of the weight block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getWeightBlockArray()

```

Sequential:getWeightBlockArray(): {FunctionBlockObject}

```

#### Returns:

* WeightBlockArray: An array containing all the weight blocks. The first function block in the array represents the first weight block.

### getInputBlockByIndex()

```

ComputationalGraph:getInputBlockByIndex(index): FunctionBlockObject

```

#### Parameters:

* index: The index of the input block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getInputBlockArray()

```

ComputationalGraph:getInputBlockArray(): {FunctionBlockObject}

```

#### Returns:

* FunctionBlockArray: An array containing all the function blocks. The first function block in the array represents the first function block.

### getOutputBlockByIndex()

```

ComputationalGraph:getOutputBlockByIndex(index): FunctionBlockObject

```

#### Parameters:

* index: The index of the output block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getOutputBlockArray()

```

ComputationalGraph:getOutputBlockArray(): {FunctionBlockObject}

```

#### Returns:

* FunctionBlockArray: An array containing all the function blocks. The first function block in the array represents the first function block.

#### Returns

* weightTensorArray: An array containing all the weight tensors from the weight blocks. The first tensor in the array represents the weight tensor for the first weight block.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)
