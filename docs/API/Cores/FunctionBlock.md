# [API Reference](../../API.md) - [Cores](../Cores.md) - FunctionBlock

## Constructors

### new()

```

FunctionBlock.new(): FunctionBlockObject

```

Returns:

* FunctionBlock: The generated function block object.

## Functions

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

* transformedInputTensor: The tensor that is transformed by the function block.

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

### setNextFunctionBlock()

```

FunctionBlock:setNextFunctionBlock(nextFunctionBlockArrayIndex: number, NextFunctionBlock: FunctionBlockObject)

```

#### Parameters:

* nextFunctionBlockArrayIndex: The index where the next function block will be placed in the next function block array.

* NextFunctionBlock: The next function block that is linked with the current function block.

### getNextFunctionBlock()

```

FunctionBlock:getNextFunctionBlock(nextFunctionBlockArrayIndex: number): FunctionBlockObject

```

#### Parameters:

* nextFunctionBlockArrayIndex: The index where the next function block is located in the next function block array.

#### Returns:

* NextFunctionBlock: The next function block that is linked with the current function block.

### setPreviousFunctionBlock()

```

FunctionBlock:setPreviousFunctionBlock(previousFunctionBlockArrayIndex: number, PreviousFunctionBlock: FunctionBlockObject)

```

#### Parameters:

* previousFunctionBlockArrayIndex: The index where the previous function block will be placed in the previous function block array.

* PreviousFunctionBlock: The previous function block that is linked with the current function block.

### getPreviousFunctionBlock()

```

FunctionBlock:getPreviousFunctionBlock(previousFunctionBlockArrayIndex: number): FunctionBlockObject

```

#### Parameters:

* previousFunctionBlockArrayIndex: The index where the previous function block is located in the previous function block array.

#### Returns:

* PreviousFunctionBlock: The previous function block that is linked with the current function block.

### removeNextFunctionBlock()

```

FunctionBlock:removeNextFunctionBlock(nextFunctionBlockArrayIndex: number)

```

#### Parameters:

* nextFunctionBlockArrayIndex: The index where the next function block is located in the next function block array.

### removePreviousFunctionBlock()

```

FunctionBlock:removePreviousFunctionBlock(previousFunctionBlockArrayIndex: number)

```

#### Parameters:

* previousFunctionBlockArrayIndex: The index where the previous function block is located in the previous function block array.

### clearAllNextFunctionBlocks()

```

FunctionBlock:clearAllNextFunctionBlocks()

```

### clearAllPreviousFunctionBlocks()

```

FunctionBlock:clearAllPreviousFunctionBlocks()

```

### setInputTensor()

```

FunctionBlock:setInputTensor(inputTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters

* inputTensor: The input tensor to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getInputTensor()

```

FunctionBlock:getInputTensor(doNotDeepCopy: boolean): tensor

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the input tensor.

#### Returns:

* inputTensor: The input tensor that is stored in the function block.

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

### setFirstDerivativeTensor()

```

FunctionBlock:setFirstDerivativeTensor(firstDerivativeTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters

* firstDerivativeTensor: The first derivative tensor to be stored into the function block.

* doNotDeepCopy: Whether or not to deep copy the input tensor.

### getFirstDerivativeTensor()

```

FunctionBlock:getFirstDerivativeTensor(doNotDeepCopy: boolean): tensor

```

#### Parameters

* doNotDeepCopy: Whether or not to deep copy the first derivative tensor.

#### Returns:

* firstDerivativeTensor: The first derivative tensor that is stored in the function block.

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

### setSaveInputTensor()

```

FunctionBlock:setSaveInputTensor(option: boolean)

```

#### Parameters:

* option: Set whether or not the input tensor is saved inside the function block.

### getSaveInputTensor()

```

FunctionBlock:getSaveInputTensor(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not the input tensor is saved inside the function block.

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

### setSaveFirstDerivativeTensor()

```

FunctionBlock:setSaveFirstDerivativeTensor(option: boolean)

```

#### Parameters:

* option: Set whether or not the first derivative tensor is saved inside the function block.

### getSaveFirstDerivativeTensor()

```

FunctionBlock:getSaveFirstDerivativeTensor(): boolean

```

#### Returns:

* option: Returns a boolean value indicating whether or not the transformed tensor is saved inside the function block.

### waitForInputTensor()

```

FunctionBlock:waitForInputTensor(doNotDeepCopy, waitDuration): tensor

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the input tensor.

*  waitDuration: The duration to wait for the input tensor to be stored into the function block before timeout.

#### Returns:

* inputTensor: The input tensor that is stored in the function block.

### waitForTransformedTensor()

```

FunctionBlock:waitForTransformedTensor(doNotDeepCopy, waitDuration): tensor

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the transformed tensor.

*  waitDuration: The duration to wait for the transformed tensor to be stored into the function block before timeout.

#### Returns:

* transformedTensor: The transformed tensor that is stored in the function block.

### waitForFirstDerivativeTensor()

```

FunctionBlock:waitForFirstDerivativeTensor(doNotDeepCopy, waitDuration): tensor

```

#### Parameters:

* doNotDeepCopy: Whether or not to deep copy the first derivative tensor.

* waitDuration: The duration to wait for the first derivative tensor to be stored into the function block before timeout.

#### Returns:

* firstDerivativeTensor: The first derivative tensor that is stored in the function block.
