# [API Reference](../../API.md) - [EncodingBlocks](../EncodingBlocks.md) - LabelEncoding

## Constructors

### new()

Creates a new encoding block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

LabelEncoding.new({valueDictionary: {any}}): EncodingBlockObject

```

Parameters:

* valueDictionary: The value dictionary to be used to convert keys stored in the tensor to values for label encoding tensor.

#### Returns:

* EncodingBlock: The generated encoding block object.

## Functions

### setParameters()

```

LabelEncoding:setParameters({valueDictionary: {}})

```

#### Parameters:

* valueDictionary: The value dictionary to be used to convert keys stored in the tensor to values for label encoding tensor.

## Inherited From

* [BaseCompressionBlock](BaseCompressionBlock.md)
