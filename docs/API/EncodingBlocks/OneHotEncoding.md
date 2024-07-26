# [API Reference](../../API.md) - [EncodingBlocks](../EncodingBlocks.md) - OneHotEncoding

## Constructors

### new()

Creates a new encoding block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

OneHotEncoding.new({finalDimensionSize: number, oneHotEncodingMode: string, indexDictionary: {any}}): EncodingBlockObject

```

Parameters:

* finalDimensionSize: The final dimension size for the transformed tensor. It is equivalent to the number of labels that are available in the data.

* oneHotEncodingMode: The encoding mode to be used by the one hot encoding block. Available options are:

	* Index (Default)

	* Key

* indexDictionary: The index dictionary to be used to convert keys stored in the tensor to one hot encoding tensor. Must be given if using the "Key" one hot encoding mode.

#### Returns:

* EncodingBlock: The generated encoding block object.

## Functions

### setParameters()

```

OneHotEncoding:setParameters({finalDimensionSize: number, oneHotEncodingMode: string, indexDictionary: {}})

```

#### Parameters:

* finalDimensionSize: The final dimension size for the transformed tensor. It is equivalent to the number of labels that are available in the data.

* oneHotEncodingMode: The encoding mode to be used by the one hot encoding block. Available options are:

	* Index

	* Key

* indexDictionary: The index dictionary to be used to convert keys stored in the tensor to one hot encoding tensor. Must be given if using the "Key" one hot encoding mode.

## Inherited From

* [BaseCompressionBlock](BaseCompressionBlock.md)
