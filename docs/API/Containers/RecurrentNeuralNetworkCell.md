# [API Reference](../../API.md) - [Containers](../Containers.md) - RecurrentNeuralNetworkCell (RNNCell)

## Constructors

### new()

Creates a new cost function object.

```

RecurrentNeuralNetworkCell.new({inputDimensionSize: number, hiddenDimensionSize: number, learningRate: number, ClassesList: {any}, cutOffValue: number}): ContainerObject

```

#### Parameters:

* inputDimensionSize: The number of input features that will be accepted by the recurrent neural network cell.

* hiddenDimensionSize: The number of hidden features that will be generated by the recurrent neural network cell.

* learningRate: The learning rate that will be used by all the weight blocks that are stored in this recurrent neural network cell.

* ClassesList: The classes to be predicted by the container. Currently, it will generate prediction based on the last dimension. [Default: {}]

* cutOffValue: The cutOffValue that classifies an output between two classes. [Default: 0]

#### Returns:

* Container: The generated container object.

## Functions

### clearAllStoredTensors()

```

RecurrentNeuralNetworkCell:clearAllStoredTensors()

```

## Inherited From

* [BaseContainer](../BaseContainer.md)
