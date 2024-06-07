# [API Reference](../../API.md) - [Containers](../Containers.md) - Sequential

## Constructors

### new()

Creates a new cost function object.

```

Sequential.new(): ContainerObject

```

#### Returns:

* Container: The generated container object.

## Functions

### setMultipleFunctionBlocks()

```

Sequential:setMultipleFunctionBlocks(...: FunctionBlock)

```

Parameters:

* FunctionBlock: The function blocks to be added to the sequential container.

### forwardPropagate()

```

Sequential:forwardPropagate(featureTensor: tensor): tensor

```

#### Parameters:

* featureTensor: The feature tensor to be used as an input.

#### Returns:

* generatedLabelTensor: The generated label tensor produced from passing across the function blocks inside the container object.

### calculateWeightLossArray()

```

Sequential:calculateWeightLossTensorArray(lossTensor: tensor): {tensor}

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

### backPropagate()

```

Sequential:backPropagate(lossTensor: tensor, numberOfData: number)


```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

* numberOfData: The value to divide with the weight loss tensors.

### clearAllStoredTensorsFromAllFunctionBlocks()

```

Sequential:clearAllStoredTensorsFromAllFunctionBlocks()

```

### detachAllFunctionBlocks()

```

Sequential:detachAllFunctionBlocks()

```

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

#### Returns

* weightTensorArray: An array containing all the weight tensors from the weight blocks. The first tensor in the array represents the weight tensor for the first weight block.

### getFunctionBlock()

```

Sequential:getFunctionBlock(index)

```

#### Parameters:

* index: The index of the function block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getFunctionBlockArray()

```

Sequential:getFunctionBlockArray()

```

#### Returns:

* FunctionBlockArray: An array containing all the function blocks. The first function block in the array represents the first function block.

## Inherited From:

* [BaseInstance](../Cores/BaseInstance.md)
