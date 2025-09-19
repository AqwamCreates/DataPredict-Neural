# [API Reference](../../API.md) - [Containers](../Containers.md) - Sequential

## Constructors

### new()

Creates a new cost function object.

```

Sequential.new({ClassesList: {any}, cutOffValue: number}): ContainerObject

```

#### Parameters:

* ClassesList: The classes to be predicted by the container. Currently, it will generate prediction based on the last dimension. [Default: {}]

* cutOffValue: The cutOff Value that classifies an output between two classes. [Default: 0]

#### Returns:

* Container: The generated container object.

## Functions

### setMultipleFunctionBlocks()

```

Sequential:setMultipleFunctionBlocks(...: FunctionBlockObject)

```

Parameters:

* FunctionBlock: The function blocks to be added to the sequential container.

### detachAllFunctionBlocks()

```

Sequential:detachAllFunctionBlocks()

```

### getFunctionBlockByIndex()

```

Sequential:getFunctionBlockByIndex(index): FunctionBlockObject

```

#### Parameters:

* index: The index of the function block.

#### Returns:

* FunctionBlock: A function block from the specified index.

### getFunctionBlockArray()

```

Sequential:getFunctionBlockArray(): {FunctionBlockObject}

```

#### Returns:

* FunctionBlockArray: An array containing all the function blocks. The first function block in the array represents the first function block.

## Inherited From

* [BaseContainer](BaseContainer.md)
