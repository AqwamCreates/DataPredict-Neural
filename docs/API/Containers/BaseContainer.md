# [API Reference](../../API.md) - [Containers](../Containers.md) - Sequential

## Constructors

### new()

Creates a new container object.

```

BaseContainer.new({ClassesList: {any}, cutOffValue: , timeDependent: boolean, parallelGrad}): ContainerObject

```

#### Returns:

* Container: The generated container object.

## Functions

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

### calculateWeightLossTensorArray()

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

### fastBackwardPropagate()

Performs gradient descent after immediately receiving the weight loss tensor for that particular weight block.

```

Sequential:fastBackwardPropagate(lossTensor: tensor)

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

### slowBackwardPropagate()

Performs gradient descent after all weight loss tensors are received from all weight blocks.

```

Sequential:fastBackwardPropagate(lossTensor: tensor)

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

### backwardPropagate()

A wrapper function that switches between fastBackwardPropagate() and slowBackwardPropagate().

```

Sequential:backwardPropagate(lossTensor: tensor)

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

#### Returns

* weightTensorArray: An array containing all the weight tensors from the weight blocks. The first tensor in the array represents the weight tensor for the first weight block.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)
