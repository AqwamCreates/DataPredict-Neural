# [API Reference](../../API.md) - [Containers](../Containers.md) - ComputationalGraph

## Constructors

### new()

Creates a new cost function object.

```

ComputationalGraph.new(): ContainerObject

```

#### Returns:

* Container: The generated container object.

## Functions

### setMultipleWeightBlocks()

```

ComputationalGraph:setMultipleWeightBlocks(...: FunctionBlock)

```

Parameters:

* WeightBlock: The weight blocks to be added to the graph container. Order matters.

### setMultipleWeightBlocks()

```

ComputationalGraph:setMultipleInputBlocks(...: FunctionBlock)

```

Parameters:

* Function: The function blocks to be added to the graph container. Order matters.

### setMultipleOutputBlocks()

```

ComputationalGraph:setMultipleOutputBlocks(...: FunctionBlock)

```

Parameters:

* Function: The function blocks to be added to the graph container. Order matters.

### forwardPropagate()

```

ComputationalGraph:forwardPropagate(featureTensorArray: {tensor}): {tensor}

```

#### Parameters:

* featureTensorArray: The feature tensor array to be used as an input. The array index determines which input block the feature tensor will pass through. For example, the feature tensor at index 1 will be used as input for the 1st input block set inside the setMultipleInputBlocks() function.

#### Returns:

* generatedLabelTensorArray: The generated label tensor array produced from passing across the function blocks inside the container object. The array index determines which output block the generated label tensor was produced from. For example, the generated label tensor at index 1 is produced from the 1st output block set inside the setMultipleOutputBlocks() function.


### calculateWeightLossArray()

```

ComputationalGraph:calculateWeightLossTensorArray(lossTensorArray: {tensor}): {tensor}

```

#### Parameters:

* lossTensorArray: The loss tensor array to be used for calculating the weights of the function blocks. The array index determines which input block the loss tensor will pass through. For example, the loss tensor at index 1 will be used as input for the 1st output block set inside the setMultipleOutputBlocks() function.


#### Returns:

* weightLossTensorArray: An array containing the weight loss tensors. The first tensor in the array represents the weight loss tensor for the first weight block.

### gradientDescent()

```

ComputationalGraph:gradientDescent(weightLossTensorArray: {tensor}, numberOfData: number)

```

#### Parameters:

* weightLossTensorArray: An array containing the weight loss tensors. The first tensor in the array represents the weight loss tensor for the first weight block.

* numberOfData: The value to divide with the weight loss tensors.

### backPropagate()

```

ComputationalGraph:backPropagate(lossTensorArray: {tensor}, numberOfData: number)

```

#### Parameters:

* lossTensor: The loss tensor array to be used for calculating the weights of the function blocks.

* numberOfData: The value to divide with the weight loss tensors.

### clearAllStoredTensorsFromAllFunctionBlocks()

```

ComputationalGraph:clearAllStoredTensorsFromAllFunctionBlocks()

```

### setWeightTensorArray()

```

ComputationalGraph:setWeightTensorArray(weightTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters:

* weightTensorArray: An array containing all the weight tensors to be set to individual weight blocks. The first tensor in the array represents the weight tensor for the first weight block.

* doNotDeepCopy: Set whether or not to deep copy the weight tensors in the weight array.

### getWeightTensorArray()

```

ComputationalGraph:getWeightTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the weight tensors from the weight blocks.

#### Returns

* weightTensorArray: An array containing all the weight tensors from the weight blocks. The first tensor in the array represents the weight tensor for the first weight block.

### getFunctionBlockByIndex()

```

ComputationalGraph:getWeightBlockByIndex(index): WeightBlock

```

#### Parameters:

* index: The index of the weight block.

#### Returns:

* WeightBlock: A weight block from the specified index.

### getWeightBlockArray()

```

ComputationalGraph:getWeightBlockArray(): {WeightBlock}

```

#### Returns:

* WeightBlockArray: An array containing all the weight blocks. The first function block in the array represents the first weight block.

### getInputBlockByIndex()

```

ComputationalGraph:getInputBlockByIndex(index): FunctionBlock

```

#### Parameters:

* index: The index of the weight block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getInputBlockArray()

```

ComputationalGraph:getInputBlockArray(): {FunctionBlock}

```

#### Returns:

* FunctionBlockArray: An array containing all the function blocks. The first function block in the array represents the first function block.

### getOutputBlockByIndex()

```

ComputationalGraph:getOutputBlockByIndex(index): FunctionBlock

```

#### Parameters:

* index: The index of the weight block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getOutputBlockArray()

```

ComputationalGraph:getOutputBlockArray(): {FunctionBlock}

```

#### Returns:

* FunctionBlockArray: An array containing all the function blocks. The first function block in the array represents the first function block.

## Inherited From:

* [BaseInstance](../Cores/BaseInstance.md)
