# [API Reference](../../API.md) - [EncodingBlocks](../EncodingBlocks.md) - PositionalEncoding

## Constructors

### new()

Creates a new encoding block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

PositionalEncoding.new({sequenceLength: number, nValue: number}): EncodingBlockObject

```

Parameters:

* sequenceLength: The length of the sequence.

* nValue: A user defined value for tuning.

#### Returns:

* EncodingBlock: The generated encoding block object.

## Functions

### setParameters()

```

PositionalEncoding:setParameters({finalDimensionSize: number, nValue: number})

```

#### Parameters:

* sequenceLength: The length of the sequence.

* nValue: A user defined value for tuning.

## Inherited From

* [BaseEncodingBlock](BaseEncodingBlock.md)

## References

[A Gentle Introduction to Positional Encoding in Transformer Models, Part 1](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
