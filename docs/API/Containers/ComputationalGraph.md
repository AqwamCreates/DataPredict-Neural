# [API Reference](../../API.md) - [Containers](../Containers.md) - ComputationalGraph

## Constructors

### new()

Creates a new cost function object.

```

ComputationalGraph.new({ClassesListArray: {{any}}, cutOffValueArray: {number}}): ContainerObject

```

#### Parameters:

* ClassesListArray: The classes to be predicted by the container. Currently, it will generate prediction based on the last dimension. [Default: {}]

* cutOffValueArray: The cutOff Value that classifies an output between two classes. [Default: 0]

#### Returns:

* Container: The generated container object.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)
