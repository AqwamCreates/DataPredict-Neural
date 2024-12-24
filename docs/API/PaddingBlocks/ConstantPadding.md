# [API Reference](../../API.md) - [PaddingBlocks](../PaddingBlocks.md) - ConstantPadding (ConstantPad)

## Constructors

### new()

Creates a new padding block object. If there are no parameters given for that particular argument, then that argument will use default value.

```

ConstantPadding.new({headPaddingDimensionSizeArray: {number}, tailPaddingDimensionSizeArray: {number}, value: number}): PaddingBlockObject

```

Parameters:

* headPaddingDimensionSizeArray: The dimension size array for padding to be placed in front of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* tailPaddingDimensionSizeArray: The dimension size array for padding to be placed at the back of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* value: The value to fill the padding with.

#### Returns:

PaddingBlockObject: The generated padding block object.

## Inherited From

* [BasePaddingBlock](BasePaddingBlock.md)
