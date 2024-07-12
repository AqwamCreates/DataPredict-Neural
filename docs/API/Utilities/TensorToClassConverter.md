# [API Reference](../../API.md) - [Utilities](../Utilities.md) - TensorToClassConverter

TensorToClassConverter is used to convert a tensor to class.

## Constructors

### new()

Creates a new tensor to class converter object.

```

TensorToClassConverter.new(classesList: {any}): TensorToClassConverterObject

```

#### Parameters:

* classesList: A list of classes. The index of the class relates to the index of the tensor at last dimension. For example, {3, 1} means that the first index of the last dimension represent the class label "3", and the second index of the last dimension represent the class label "1".

#### Returns:

* TensorToClassConverterObject: An object that allows the conversion of tensor to classes.

## Functions:

### convert()

```

TensorToClassConverter:convert(tensorToBeConverted: tensor): tensor

```

#### Parameters:

* tensorToBeConverted: The tensor to be converted.

#### Returns:

* classTensor: The tensor containing the class values. The shape of the tensor is the same as "tensorToBeConverted", except that the size at the final dimension is equal to 1.

### setClassesList()

```

TensorToClassConverter:setClassesList(classesList: {any})

```

#### Parameters:

* classesList: A list of classes. The index of the class relates to the index of the tensor at last dimension. For example, {3, 1} means that the first index of the last dimension represent the class label "3", and the second index of the last dimension represent the class label "1".

### getClassesList()

```

TensorToClassConverter:getClassesList(): {any}

```

#### Returns:

* classesList:  A list of classes. The index of the class relates to the index of the tensor at last dimension. For example, {3, 1} means that the first index of the last dimension represent the class label "3", and the second index of the last dimension represent the class label "1".

## Inherited From:

[BaseInstance](../Cores/BaseInstance.md)