# [API Reference](../../API.md) - [Cores](../Cores.md) - BaseFunctionBlock

## Constructors

### new()

```

BaseFunctionBlock.new(): BaseFunctionBlockObject

```

Returns:

* BaseFunctionBlock: The generated function block object.

## Functions

### pullAllInputTensors()

Pulls all the tensors from the previous function blocks and pass them to the transform() function. Commonly used when trying to get the output of the current function block after linking to the previous function block once the data has already been passed through the whole model without the current function block in it.

```

BaseFunctionBlock:pullAllInputTensors()

```

### linkForward()

```

BaseFunctionBlock:linkForward(NextBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: The next function block to be linked with the current function block.

### multipleLinkForward()

```

BaseFunctionBlock:multipleLinkForward(...: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: A variable number of next function blocks to be linked with the current function block.

### linkBackward()

```

BaseFunctionBlock:linkBackward(PreviousBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: The previous function block to be linked with the current function block.

### multipleLinkBackward()

```

BaseFunctionBlock:multipleLinkBackward(...: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: A variable number of previous function blocks to be linked with the current function block.

### unlinkForward()

```

BaseFunctionBlock:unlinkForward(NextBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: The next function block to be unlinked from the current function block.

### multipleUnlinkForward()

```

BaseFunctionBlock:multipleUnlinkForward(...: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: A variable number of next function blocks to be unlinked from the current function block.

### unlinkBackward()

```

BaseFunctionBlock:unlinkBackward(PreviousBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: The previous function block to be unlinked from the current function block.

### multipleUnlinkBackward()

```

BaseFunctionBlock:multipleUnlinkBackward(...: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: A variable number of previous function blocks to be unlinked from the current function block.

### setFunction()

```

BaseFunctionBlock:setFunction(Function: Function)

```

#### Parameters:

* Function: The function to be used when calling the transform() function.

### setChainRuleFirstDerivativeFunction()

```

BaseFunctionBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction: Function)

```

#### Parameters:

* ChainRuleFirstDerivativeFunction: The chain rule first derivative function to be used when calling the differentiate() function.

### setFirstDerivativeFunction()

```

BaseFunctionBlock:setFirstDerivativeFunction(FirstDerivativeFunction: Function)

```

#### Parameters:

* FirstDerivativeFunction: The first derivative function to be used when calling the differentiate() function.

### transform()

```

BaseFunctionBlock:transform(inputTensor: tensor): tensor

```

#### Parameters:

* inputTensor: The tensor to be transformed by the function block.

#### Returns:

* transformedTensor: The tensor that is transformed by the function block.

### differentiate()

```

BaseFunctionBlock:differentiate(initialFirstDerivativeTensor: tensor, transformedTensor: tensor, inputTensor: tensor): tensor

```

#### Parameters:

* initialFirstDerivativeTensor: The tensor to multiply with the first derivative tensor calculated by the function block. If not given, it will use a tensor containing values of 1.

* transformedTensor: The tensor that is transformed by the function block. If not given, it will used stored transformed tensor or calculate a new transformed tensor.

* inputTensor: The tensor to be transformed by the function block. If not given, it will used stored input tensor.

#### Returns:

* firstDerivativeTensor: The tensor that is transformed by the function block using first derivative function.

### addNextBaseFunctionBlock()

```

BaseFunctionBlock:addNextBaseFunctionBlock(NextBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: The next function block to be linked with the current function block.

### addNextBaseFunctionBlock()

```

BaseFunctionBlock:addPreviousBaseFunctionBlock(PreviousBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: The previous function block to be linked with the current function block.

### addMultipleNextBaseFunctionBlocks()

```

BaseFunctionBlock:addMultipleNextBaseFunctionBlocks(...: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: The next function block to be linked with the current function block.

### addMultiplePreviousBaseFunctionBlocks()

```

BaseFunctionBlock:addMultiplePreviousBaseFunctionBlocks(...: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: The previous function block to be linked with the current function block.

### setChainRuleFirstDerivativeFunctionRequiresTransformedTensor()

```

BaseFunctionBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(option)

```

#### Parameters:

* option: Set whether or not the chain rule first derivative function requires the transformed input tensor.

### getChainRuleFirstDerivativeFunctionRequiresTransformedTensor()

```

BaseFunctionBlock:getChainRuleFirstDerivativeFunctionRequiresTransformedTensor(option: boolean)

```

#### Returns:

* option: Set whether or not the chain rule first derivative function requires the transformed input tensor.

### setNextBaseFunctionBlockByIndex()

```

BaseFunctionBlock:setNextBaseFunctionBlockByIndex(nextBaseFunctionBlockArrayIndex: number, NextBaseFunctionBlock: BaseFunctionBlockObject)

```

### setFirstDerivativeFunctionRequiresTransformedTensor()

```

BaseFunctionBlock:setFirstDerivativeFunctionRequiresTransformedTensor(option)

```

#### Parameters:

* option: Set whether or not the first derivative function requires the transformed input tensor.

### getFirstDerivativeFunctionRequiresTransformedTensor()

```

BaseFunctionBlock:getFirstDerivativeFunctionRequiresTransformedTensor(option: boolean)

```

#### Returns:

* option: Set whether or not the first derivative function requires the transformed input tensor.

### setNextBaseFunctionBlockByIndex()

```

BaseFunctionBlock:setNextBaseFunctionBlockByIndex(nextBaseFunctionBlockArrayIndex: number, NextBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* nextBaseFunctionBlockArrayIndex: The index where the next function block will be placed in the next function block array.

* NextBaseFunctionBlock: The next function block that is linked with the current function block.

### getNextBaseFunctionBlockByIndex()

```

BaseFunctionBlock:getNextBaseFunctionBlockByIndex(nextBaseFunctionBlockArrayIndex: number): BaseFunctionBlockObject

```

#### Parameters:

* nextBaseFunctionBlockArrayIndex: The index where the next function block is located in the next function block array.

#### Returns:

* NextBaseFunctionBlock: The next function block that is linked with the current function block.

### setPreviousBaseFunctionBlockByIndex()

```

BaseFunctionBlock:setPreviousBaseFunctionBlockByIndex(previousBaseFunctionBlockArrayIndex: number, PreviousBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* previousBaseFunctionBlockArrayIndex: The index where the previous function block will be placed in the previous function block array.

* PreviousBaseFunctionBlock: The previous function block that is linked with the current function block.

### getPreviousBaseFunctionBlockByIndex()

```

BaseFunctionBlock:getPreviousBaseFunctionBlockByIndex(previousBaseFunctionBlockArrayIndex: number): BaseFunctionBlockObject

```

#### Parameters:

* previousBaseFunctionBlockArrayIndex: The index where the previous function block is located in the previous function block array.

#### Returns:

* PreviousBaseFunctionBlock: The previous function block that is linked with the current function block.

### removeNextBaseFunctionBlock()

```

BaseFunctionBlock:removeNextBaseFunctionBlock(NextBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* NextBaseFunctionBlock: The next function block to remove from the next function block array.

### removePreviousBaseFunctionBlock()

```

BaseFunctionBlock:removePreviousBaseFunctionBlock(PreviousBaseFunctionBlock: BaseFunctionBlockObject)

```

#### Parameters:

* PreviousBaseFunctionBlock: The previous function block to remove from the previous function block array.

### removeNextBaseFunctionBlockByIndex()

```

BaseFunctionBlock:removeNextBaseFunctionBlockByIndex(nextBaseFunctionBlockArrayIndex: number)

```

#### Parameters:

* nextBaseFunctionBlockArrayIndex: The index where the next function block is located in the next function block array.

### removePreviousBaseFunctionBlockByIndex()

```

BaseFunctionBlock:removePreviousBaseFunctionBlockByIndex(previousBaseFunctionBlockArrayIndex: number)

```

#### Parameters:

* previousBaseFunctionBlockArrayIndex: The index where the previous function block is located in the previous function block array.

### clearNextBaseFunctionBlockArray()

```

BaseFunctionBlock:clearNextBaseFunctionBlockArray()

```

### clearPreviousBaseFunctionBlockArray()

```

BaseFunctionBlock:clearPreviousBaseFunctionBlockArray()

```

### setInputTensor()

```

BaseFunctionBlock:setInputTensorArray(inputTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters

* inputTensorArray: An array containing all the input tensors to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getInputTensorArray()

```

BaseFunctionBlock:getInputTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the input tensor.

#### Returns:

* inputTensorArray: An array containing all the input tensors that is stored in the function block.

### setTransformedTensor()

```

BaseFunctionBlock:setTransformedTensor(transformedTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters

* transformedTensor: The transformed tensor to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

### getTransformedTensor()

```

BaseFunctionBlock:getTransformedTensor(doNotDeepCopy: boolean): tensor

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

#### Returns:

* transformedTensor: The transformed tensor that is stored in the function block.

### setTotalPartialFirstDerivativeTensorArray()

```

BaseFunctionBlock:setTotalChainRuleFirstDerivativeTensorArray(chainRuleFirstDerivativeTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters

* chainRuleFirstDerivativeTensorArray: An array containing all the total of chain rule first derivative tensors to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getTotalChainRuleFirstDerivativeTensorArray()

```

BaseFunctionBlock:getTotalChainRuleFirstDerivativeTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the total of partial first derivative tensor array.

#### Returns:

* chainRuleFirstDerivativeTensorArray: An array containing all the total of chain rule first derivative tensors that is stored in the function block.

### setFirstDerivativeTensorArray()

```

BaseFunctionBlock:setTotalFirstDerivativeTensorArray(firstDerivativeTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters

* firstDerivativeTensorArray: An array containing all the total of first derivative tensors to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getTotalFirstDerivativeTensorArray()

```

BaseFunctionBlock:getTotalFirstDerivativeTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the first derivative tensor.

#### Returns:

* totalFirstDerivativeTensorArray: An array containing all the total of first derivative tensors that is stored in the function block.

### getNextBaseFunctionBlockArray()

```

BaseFunctionBlock:getNextBaseFunctionBlockArray(): {BaseFunctionBlock}

```

#### Returns:

* NextBaseFunctionBlockArray: An array containing next function blocks. The first next function block in the array represents the first next function block connected with the current function block.

### getPreviousBaseFunctionBlockArray()

```

BaseFunctionBlock:getPreviousBaseFunctionBlockArray(): {BaseFunctionBlock}

```

#### Returns:

* PreviousBaseFunctionBlockArray: An array containing previous function blocks. The first next function block in the array represents the previous next function block connected with the current function block.

### setWaitForAllInitialPartialFirstDerivativeTensors()

```

BaseFunctionBlock:setWaitForAllInitialPartialFirstDerivativeTensors(option: boolean)

```

#### Parameters:

* option: Set whether or not the current function block must wait for all initial partial first derivative from all function block paths.

### getSaveInputTensorArray()

```

BaseFunctionBlock:getWaitForAllInitialPartialFirstDerivativeTensors(): boolean

```

#### Returns:

* option: Returns a boolean value that determines if the current function block must wait for all initial partial first derivative from all function block paths.

### setSaveInputTensor()

```

BaseFunctionBlock:setSaveInputTensorArray(option: boolean)

```

#### Parameters:

* option: Set whether or not if all the input tensors are saved inside the function block.

### getSaveInputTensorArray()

```

BaseFunctionBlock:getSaveInputTensorArray(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not if all the input tensors are saved inside the function block.

### setSaveTransformedTensor()

```

BaseFunctionBlock:setSaveTransformedTensor(option: boolean)

```

#### Parameters:

* option: Set whether or not the transformed tensor is saved inside the function block.

### getSaveTransformedTensor()

```

BaseFunctionBlock:getSaveTransformedTensor(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not the transformed tensor is saved inside the function block.

### setSaveTotalPartialFirstDerivativeTensorArray()

```

BaseFunctionBlock:setSaveTotalChainRuleFirstDerivativeTensorArray(option: boolean)

```

#### Parameters:

* option: Set whether or not if all the total of partial first derivative tensors are saved inside the function block.

### getSaveTotalPartialFirstDerivativeTensorArray()

```

BaseFunctionBlock:getSaveTotalChainRuleFirstDerivativeTensorArray(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not if all the total of partial first derivative tensors are saved inside the function block.

### setSaveTotalFirstDerivativeTensorArray()

```

BaseFunctionBlock:setSaveTotalFirstDerivativeTensorArray(option: boolean)

```

#### Parameters:

* option: Set whether or not if all the total of first derivative tensors are saved inside the function block.

### getSaveTotalFirstDerivativeTensorArray()

```

BaseFunctionBlock:getSaveTotalFirstDerivativeTensorArray(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not if all the total of first derivative tensors are saved inside the function block.

### waitForInputTensorArray()

```

BaseFunctionBlock:waitForInputTensorArray(doNotDeepCopy, waitDuration): {tensor}

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the input tensor.

* waitDuration: The duration to wait for the input tensor to be stored into the function block before timeout.

#### Returns:

* inputTensor: An array containing all the input tensor that is stored in the function block.

### waitForTransformedTensor()

```

BaseFunctionBlock:waitForTransformedTensor(doNotDeepCopy, waitDuration): tensor

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

* waitDuration: The duration to wait for the transformed tensor to be stored into the function block before timeout.

#### Returns:

* transformedTensor: The transformed tensor that is stored in the function block.

### waitForTotalPartialFirstDerivativeTensorArray()

```

BaseFunctionBlock:waitForTotalChainRuleFirstDerivativeTensorArray(doNotDeepCopy, waitDuration): {tensor}

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the partial first derivative tensor array.

* waitDuration: The duration to wait for the partial first derivative tensor array to be stored into the function block before timeout.

#### Returns:

* totalChainRuleFirstDerivativeTensorArray: An array containing all the total of chain rule first derivative tensors that is stored in the function block.

### waitForTotalFirstDerivativeTensorArray()

```

BaseFunctionBlock:waitForTotalFirstDerivativeTensorArray(doNotDeepCopy, waitDuration): {tensor}

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the first derivative tensor array.

* waitDuration: The duration to wait for the first derivative tensor array to be stored into the function block before timeout.

#### Returns:

* totalFirstDerivativeTensorArray: An array containing all the total of first derivative tensors that is stored in the function block.

## Inherited From

* [BaseInstance](BaseInstance.md)
