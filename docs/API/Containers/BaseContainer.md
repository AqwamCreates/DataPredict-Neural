# [API Reference](../../API.md) - [Containers](../Containers.md) - Sequential

## Constructors

### new()

Creates a new container object.

```

BaseContainer.new({ClassesList: {any}, cutOffValue: number, timeDependent: boolean, parallelGradientDescent: boolean, WeightBlockArray: {WeightBlock}}): ContainerObject

```

#### Parameters:

* ClassesList: The classes to be predicted by the container. Currently, it will generate prediction based on the last dimension. [Default: {}]

* cutOffValue: The cutOffValue that classifies an output between two classes. [Default: 0]

* timeDependent: An indicator that the container uses datasets that are time dependent. Enabling this will cause the container to always use slower backward propagation method. [Default: False]

* parallelGradientDescent: An indicator that container will perform parallel gradient descent. See fastBackwardPropagate() function for more details.

* WeightBlockArray: An array containing all the weight blocks that will be loaded to the container.

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

Performs the gradient descent after immediately receiving the weight loss tensor for that particular weight block.

```

Sequential:fastBackwardPropagate(lossTensor: tensor)

```

#### Parameters:

* lossTensor: The loss tensor to be used for calculating the weights of the function blocks.

### slowBackwardPropagate()

Performs the gradient descent after all weight loss tensors are received from all weight blocks.

```

Sequential:slowBackwardPropagate(lossTensor: tensor)

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
