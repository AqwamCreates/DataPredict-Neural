# [API Reference](../../API.md) - [Cores](../Cores.md) - AutomaticDifferentiationTensor

## Constructors

### new()

```

AutomaticDifferentiationTensor.new(tensor: tensor, PartialDerivativeFunction: Function, inputTensorArray: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

* PartialDerivativeFunction (Optional): The partial derivative function to be multiplied with initialPartialFirstDerivativeTensor. Must supply the initialPartialFirstDerivativeTensor argument to the function.

* inputTensorArray (Optional): The input tensor objects that was used to generate the current tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### sin()

```

AutomaticDifferentiationTensor.sin(tensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### cos()

```

AutomaticDifferentiationTensor.cos(tensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### tan()

```

AutomaticDifferentiationTensor.tan(tensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### exponent()

```

AutomaticDifferentiationTensor.exponent(tensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### logarithm()

```

AutomaticDifferentiationTensor.logarithm(numberTensor: number/tensor/AutomaticDifferentiationTensorObject, baseTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* numberTensor: The tensor to be used by the automatic differentiation tensor object.

* baseTensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### clamp()

```

AutomaticDifferentiationTensor.clamp(tensor: tensor/AutomaticDifferentiationTensorObject, upperBoundTensor: number/tensor/AutomaticDifferentiationTensorObject upperBoundTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

### maximum()

```

AutomaticDifferentiationTensor.maximum(tensor1: number/tensor/AutomaticDifferentiationTensorObject, tensor2: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### minimum()

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

```

AutomaticDifferentiationTensor.minimum(tensor1: number/tensor/AutomaticDifferentiationTensorObject, tensor2: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

## Functions

### add()

```

AutomaticDifferentiationTensor:add(AutomaticDifferentiationTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### subtract()

```

AutomaticDifferentiationTensor:subtract(AutomaticDifferentiationTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### multiply()

```

AutomaticDifferentiationTensor:multiply(AutomaticDifferentiationTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### divide()

```

AutomaticDifferentiationTensor:divide(AutomaticDifferentiationTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### power()

```

AutomaticDifferentiationTensor:power(AutomaticDifferentiationTensor: number/tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The exponent tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### sum()

```

AutomaticDifferentiationTensor:sum(dimension: number): AutomaticDifferentiationTensorObject

```

#### Parameters:

* dimension: The dimension to sum across the tensor.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### dotProduct()

```

AutomaticDifferentiationTensor:dotProduct(AutomaticDifferentiationTensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.


### getTensor()

```

AutomaticDifferentiationTensor:getTensor(doNotDeepCopy: boolean): tensor

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the tensor.

#### Returns:

* tensor: The tensor generated by the automatic differentiation tensor object.

### setTensor()

```

AutomaticDifferentiationTensor:getTensor(tensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

* doNotDeepCopy: Set whether or not to deep copy the tensor.

### getTotalPartialFirstDerivativeTensor()

```

AutomaticDifferentiationTensor:getTotalPartialFirstDerivativeTensor(doNotDeepCopy: boolean): tensor

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the tensor.

#### Returns:

* totalPartialFirstDerivativeTensor: The total of partial first derivative tensor generated by the automatic differentiation tensor object.

### setTotalPartialFirstDerivativeTensor()

```

AutomaticDifferentiationTensor:setTotalPartialFirstDerivativeTensor(totalPartialFirstDerivativeTensor: tensor, doNotDeepCopy: boolean)

```

#### Parameters:

* totalPartialFirstDerivativeTensor: The total of partial first derivative tensor to be used by the automatic differentiation tensor object.

* doNotDeepCopy: Set whether or not to deep copy the tensor.

### destroy()

```

AutomaticDifferentiationTensor:destroy(areDescendantsDestroyed: boolean)

```

### Parameters:

* areDescendantsDestroyed: Set whether or not to destroy the descendants of the current automatic differentiation tensor object.
