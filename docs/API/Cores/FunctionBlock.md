# [API Reference](../../API.md) - [Cores](../Cores.md) - FunctionBlock

## Constructors

### new()

```

FunctionBlock.new(): FunctionBlockObject

```

Returns:

* FunctionBlock: The generated function block object.

## Functions

### linkForward()

```
FunctionBlock:linkForward(NextFunctionBlock: FunctionBlockObject)
```

#### Parameters:

* NextFunctionBlock: The next function block to be linked with the current function block.

### multipleLinkForward()

```
FunctionBlock:multipleLinkForward(...: FunctionBlockObject)
```

#### Parameters:

* NextFunctionBlock: A variable number of next function blocks to be linked with the current function block.

### linkBackward()

```
FunctionBlock:linkBackward(PreviousFunctionBlock: FunctionBlockObject)
```

#### Parameters:

* PreviousFunctionBlock: The previous function block to be linked with the current function block.

### multipleLinkBackward()

```
FunctionBlock:multipleLinkBackward(...: FunctionBlockObject)
```

#### Parameters:

* PreviousFunctionBlock: A variable number of previous function blocks to be linked with the current function block.

### unlinkForward()

```
FunctionBlock:unlinkForward(NextFunctionBlock: FunctionBlockObject)
```

#### Parameters:

* NextFunctionBlock: The next function block to be unlinked from the current function block.

### multipleUnlinkForward()

```
FunctionBlock:multipleUnlinkForward(...: FunctionBlockObject)
```

#### Parameters:

* NextFunctionBlock: A variable number of next function blocks to be unlinked from the current function block.

### unlinkBackward()

```
FunctionBlock:unlinkBackward(PreviousFunctionBlock: FunctionBlockObject)
```

#### Parameters:

* PreviousFunctionBlock: The previous function block to be unlinked from the current function block.

### multipleUnlinkBackward()

```
FunctionBlock:multipleUnlinkBackward(...: FunctionBlockObject)
```

#### Parameters:

* PreviousFunctionBlock: A variable number of previous function blocks to be unlinked from the current function block.

---

Let me know if you need any more details or additional methods documented!

### setFunction()

```

FunctionBlock:setFunction(Function: Function)

```

#### Parameters:

* Function: The function to be used when calling the transform() function.

### setFirstDerivativeFunction()

```

FunctionBlock:setFirstDerivativeFunction(FirstDerivativeFunction: Function)

```

#### Parameters:

* FirstDerivativeFunction: The function to be used when calling the differentiate() function.

### transform()

```

FunctionBlock:transform(inputTensor: tensor): tensor

```

#### Parameters:

* inputTensor: The tensor to be transformed by the function block.

#### Returns:

* transformedTensor: The tensor that is transformed by the function block.

### differentiate()

```

FunctionBlock:differentiate(initialFirstDerivativeTensor: tensor, transformedTensor: tensor, inputTensor: tensor): tensor

```

#### Parameters:

* initialFirstDerivativeTensor: The tensor to multiply with the first derivative tensor calculated by the function block. If not given, it will use a tensor containing values of 1.

* transformedTensor: The tensor that is transformed by the function block. If not given, it will used stored transformed tensor or calculate a new transformed tensor.

* inputTensor: The tensor to be transformed by the function block. If not given, it will used stored input tensor.

#### Returns:

* firstDerivativeTensor: The tensor that is transformed by the function block using first derivative function.

### addNextFunctionBlock()

```

FunctionBlock:addNextFunctionBlock(NextFunctionBlock: NextFunctionBlockObject)

```

#### Parameters:

* NextFunctionBlock: The next function block to be linked with the current function block.

### addNextFunctionBlock()

```

FunctionBlock:addPreviousFunctionBlock(PreviousFunctionBlock: PreviousFunctionBlockObject)

```

#### Parameters:

* PreviousFunctionBlock: The previous function block to be linked with the current function block.

### addMultipleNextFunctionBlocks()

```

FunctionBlock:addMultipleNextFunctionBlocks(...: NextFunctionBlockObject)

```

#### Parameters:

* NextFunctionBlock: The next function block to be linked with the current function block.

### addMultiplePreviousFunctionBlocks()

```

FunctionBlock:addMultiplePreviousFunctionBlocks(...: PreviousFunctionBlockObject)

```

#### Parameters:

* PreviousFunctionBlock: The previous function block to be linked with the current function block.

### setFirstDerivativeFunctionRequiresTransformedTensor()

```

FunctionBlock:setFirstDerivativeFunctionRequiresTransformedTensor(option)

```

#### Parameters:

* option: Set whether or not the first derivative function requires the transformed input tensor.

### getFirstDerivativeFunctionRequiresTransformedTensor()

```

FunctionBlock:getFirstDerivativeFunctionRequiresTransformedTensor(option: boolean)

```

#### Returns:

* option: Set whether or not the first derivative function requires the transformed input tensor.

### setNextFunctionBlockByIndex()

```

FunctionBlock:setNextFunctionBlockByIndex(nextFunctionBlockArrayIndex: number, NextFunctionBlock: FunctionBlockObject)

```

#### Parameters:

* nextFunctionBlockArrayIndex: The index where the next function block will be placed in the next function block array.

* NextFunctionBlock: The next function block that is linked with the current function block.

### getNextFunctionBlockByIndex()

```

FunctionBlock:getNextFunctionBlockByIndex(nextFunctionBlockArrayIndex: number): FunctionBlockObject

```

#### Parameters:

* nextFunctionBlockArrayIndex: The index where the next function block is located in the next function block array.

#### Returns:

* NextFunctionBlock: The next function block that is linked with the current function block.

### setPreviousFunctionBlockByIndex()

```

FunctionBlock:setPreviousFunctionBlockByIndex(previousFunctionBlockArrayIndex: number, PreviousFunctionBlock: FunctionBlockObject)

```

#### Parameters:

* previousFunctionBlockArrayIndex: The index where the previous function block will be placed in the previous function block array.

* PreviousFunctionBlock: The previous function block that is linked with the current function block.

### getPreviousFunctionBlockByIndex()

```

FunctionBlock:getPreviousFunctionBlockByIndex(previousFunctionBlockArrayIndex: number): FunctionBlockObject

```

#### Parameters:

* previousFunctionBlockArrayIndex: The index where the previous function block is located in the previous function block array.

#### Returns:

* PreviousFunctionBlock: The previous function block that is linked with the current function block.

### removeNextFunctionBlockByIndex()

```

FunctionBlock:removeNextFunctionBlockByIndex(nextFunctionBlockArrayIndex: number)

```

#### Parameters:

* nextFunctionBlockArrayIndex: The index where the next function block is located in the next function block array.

### removePreviousFunctionBlockByIndex()

```

FunctionBlock:removePreviousFunctionBlockByIndex(previousFunctionBlockArrayIndex: number)

```

#### Parameters:

* previousFunctionBlockArrayIndex: The index where the previous function block is located in the previous function block array.

### clearNextFunctionBlockArray()

```

FunctionBlock:clearNextFunctionBlockArray()

```

### clearPreviousFunctionBlockArray()

```

FunctionBlock:clearPreviousFunctionBlockArray()

```

### setInputTensor()

```

FunctionBlock:setInputTensorArray(inputTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters

* inputTensorArray: An array containing all the input tensors to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getInputTensorArray()

```

FunctionBlock:getInputTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the input tensor.

#### Returns:

* inputTensorArray: An array containing all the input tensors that is stored in the function block.

### setTransformedTensor()

```

FunctionBlock:setTransformedTensor(transformedTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters

* transformedTensor: The transformed tensor to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

### getTransformedTensor()

```

FunctionBlock:getTransformedTensor(doNotDeepCopy: boolean): tensor

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

#### Returns:

* transformedTensor: The transformed tensor that is stored in the function block.

### setTotalPartialFirstDerivativeTensorArray()

```

FunctionBlock:setTotalPartialFirstDerivativeTensorArray(partialFirstDerivativeTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters

* totalPartialFirstDerivativeTensorArray: An array containing all the total of partial first derivative tensors to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getFirstDerivativeTensorArray()

```

FunctionBlock:getTotalFirstDerivativeTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the total of partial first derivative tensor array.

#### Returns:

* firstDerivativeTensorArray: An array containing all the total of partial first derivative tensors that is stored in the function block.

### setFirstDerivativeTensorArray()

```

FunctionBlock:setTotalFirstDerivativeTensorArray(firstDerivativeTensorArray: {tensor}, doNotDeepCopy: boolean)

```

#### Parameters

* firstDerivativeTensorArray: An array containing all the total of first derivative tensors to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getTotalFirstDerivativeTensorArray()

```

FunctionBlock:getTotalFirstDerivativeTensorArray(doNotDeepCopy: boolean): {tensor}

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the first derivative tensor.

#### Returns:

* totalFirstDerivativeTensorArray: An array containing all the total of first derivative tensors that is stored in the function block.

### getNextFunctionBlockArray()

```

FunctionBlock:getNextFunctionBlockArray(): {FunctionBlock}

```

#### Returns:

* NextFunctionBlockArray: An array containing next function blocks. The first next function block in the array represents the first next function block connected with the current function block.

### getPreviousFunctionBlockArray()

```

FunctionBlock:getPreviousFunctionBlockArray(): {FunctionBlock}

```

#### Returns:

* PreviousFunctionBlockArray: An array containing previous function blocks. The first next function block in the array represents the previous next function block connected with the current function block.

### setWaitForAllInitialPartialFirstDerivativeTensors()

```

FunctionBlock:setWaitForAllInitialPartialFirstDerivativeTensors(option: boolean)

```

#### Parameters:

* option: Set whether or not the current function block must wait for all initial partial first derivative from all function block paths.

### getSaveInputTensorArray()

```

FunctionBlock:getWaitForAllInitialPartialFirstDerivativeTensors(): boolean

```

#### Returns:

* option: Returns a boolean value that determines if the current function block must wait for all initial partial first derivative from all function block paths.

### setSaveInputTensor()

```

FunctionBlock:setSaveInputTensorArray(option: boolean)

```

#### Parameters:

* option: Set whether or not if all the input tensors are saved inside the function block.

### getSaveInputTensorArray()

```

FunctionBlock:getSaveInputTensorArray(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not if all the input tensors are saved inside the function block.

### setSaveTransformedTensor()

```

FunctionBlock:setSaveTransformedTensor(option: boolean)

```

#### Parameters:

* option: Set whether or not the transformed tensor is saved inside the function block.

### getSaveTransformedTensor()

```

FunctionBlock:getSaveTransformedTensor(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not the transformed tensor is saved inside the function block.

### setSaveTotalPartialFirstDerivativeTensorArray()

```

FunctionBlock:setSaveTotalPartialFirstDerivativeTensorArray(option: boolean)

```

#### Parameters:

* option: Set whether or not if all the total of partial first derivative tensors are saved inside the function block.

### getSaveTotalPartialFirstDerivativeTensorArray()

```

FunctionBlock:getSaveTotalPartialFirstDerivativeTensorArray(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not if all the total of partial first derivative tensors are saved inside the function block.

### setSaveTotalFirstDerivativeTensorArray()

```

FunctionBlock:setSaveTotalFirstDerivativeTensorArray(option: boolean)

```

#### Parameters:

* option: Set whether or not if all the total of first derivative tensors are saved inside the function block.

### getSaveTotalFirstDerivativeTensorArray()

```

FunctionBlock:getSaveTotalFirstDerivativeTensorArray(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not if all the total of first derivative tensors are saved inside the function block.

### waitForInputTensorArray()

```

FunctionBlock:waitForInputTensorArray(doNotDeepCopy, waitDuration): {tensor}

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the input tensor.

* waitDuration: The duration to wait for the input tensor to be stored into the function block before timeout.

#### Returns:

* inputTensor: An array containing all the input tensor that is stored in the function block.

### waitForTransformedTensor()

```

FunctionBlock:waitForTransformedTensor(doNotDeepCopy, waitDuration): tensor

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

* waitDuration: The duration to wait for the transformed tensor to be stored into the function block before timeout.

#### Returns:

* transformedTensor: The transformed tensor that is stored in the function block.

### waitForTotalPartialFirstDerivativeTensorArray()

```

FunctionBlock:waitForTotalPartialFirstDerivativeTensorArray(doNotDeepCopy, waitDuration): {tensor}

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the partial first derivative tensor array.

* waitDuration: The duration to wait for the partial first derivative tensor array to be stored into the function block before timeout.

#### Returns:

* totalPartialFirstDerivativeTensorArray: An array containing all the total of partial first derivative tensor that is stored in the function block.

### waitForTotalFirstDerivativeTensorArray()

```

FunctionBlock:waitForTotalFirstDerivativeTensorArray(doNotDeepCopy, waitDuration): {tensor}

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the first derivative tensor array.

* waitDuration: The duration to wait for the first derivative tensor array to be stored into the function block before timeout.

#### Returns:

* totalFirstDerivativeTensorArray: An array containing all the total of first derivative tensor that is stored in the function block.
