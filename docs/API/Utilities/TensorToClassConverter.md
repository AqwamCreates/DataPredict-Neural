# [API Reference](../../API.md) - [Utilities](../Utilities.md) - TensorToClassConverter

TensorToClassConverter is used to convert a tensor to class.

## Constructors

### new()

Creates a new tensor to class converter object.

```

TensorToClassConverter.new(ClassesList: {any}): TensorToClassConverterObject

```

#### Parameters:

* ClassesList: A list of classes. The index of the class relates to the index of the tensor at last dimension. For example, {3, 1} means that the first index of the last dimension represent the class label "3", and the second index of the last dimension represent the class label "1".

#### Returns:

* TensorToClassConverterObject: An object that allows the conversion of tensor to classes.

## Functions:

### convert()

```

TensorToClassConverter:convert(tensor: tensor): tensor

```

#### Parameters:

* tensor: The tensor to be converted.

#### Returns:

* classTensor: The tensor containing the class values. The shape of the tensor is the same as "tensor", except that the size at the final dimension is equal to 1.

### setClassesList()

```

TensorToClassConverter:setClassesList(ClassesList: {any})

```

#### Parameters:

* ClassesList: A list of classes. The index of the class relates to the index of the tensor at last dimension. For example, {3, 1} means that the first index of the last dimension represent the class label "3", and the second index of the last dimension represent the class label "1".

### getClassesList()

```

TensorToClassConverter:getClassesList(): {any}

```

#### Returns:

* ClassesList:  A list of classes. The index of the class relates to the index of the tensor at last dimension. For example, {3, 1} means that the first index of the last dimension represent the class label "3", and the second index of the last dimension represent the class label "1".

## Inherited From

[BaseInstance](../Cores/BaseInstance.md)
