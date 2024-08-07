# [API Reference](../../API.md) - [Cores](../Cores.md) - AutomaticDifferentiationTensor

## Constructors

### new()

```

AutomaticDifferentiationTensorObject.new(tensor: tensor, PartialDerivativeFunction: Function, PreviousTensorObject1: AutomaticDifferentiationTensorObject, PreviousTensorObject2: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

* PartialDerivativeFunction (Optional): The partial derivative function to be multiplied with initialPartialFirstDerivativeTensor. Must supply the initialPartialFirstDerivativeTensor argument to the function.

* PreviousTensorObject1 (Optional): The first previous tensor object that was used to generate the current tensor object.

* PreviousTensorObject2 (Optional): The second previous tensor object that was used to generate the current tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### sin()

```

AutomaticDifferentiationTensorObject.sin(tensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### cos()

```

AutomaticDifferentiationTensorObject.cos(tensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### tan()

```

AutomaticDifferentiationTensorObject.tan(tensor: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### max()

```

AutomaticDifferentiationTensorObject.max(tensor1: tensor/AutomaticDifferentiationTensorObject, tensor2: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### min()

#### Parameters:

* tensor: The tensor to be used by the automatic differentiation tensor object.

```

AutomaticDifferentiationTensorObject.min(tensor1: tensor/AutomaticDifferentiationTensorObject, tensor2: tensor/AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

## Functions

### add()

```

AutomaticDifferentiationTensorObject:add(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### subtract()

```

AutomaticDifferentiationTensorObject:subtract(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### multiply()

```

AutomaticDifferentiationTensorObject:multiply(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### divide()

```

AutomaticDifferentiationTensorObject:divide(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### power()

```

AutomaticDifferentiationTensorObject.power(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The exponent tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### log()

```

AutomaticDifferentiationTensorObject:log(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

```

#### Parameters:

* AutomaticDifferentiationTensor: The base tensor object to be used by the automatic differentiation tensor object.

#### Returns

* AutomaticDifferentiationTensorObject: The generated automatic differentiation tensor object.

### dotProduct()

```

AutomaticDifferentiationTensorObject:dotProduct(AutomaticDifferentiationTensor: AutomaticDifferentiationTensorObject): AutomaticDifferentiationTensorObject

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
