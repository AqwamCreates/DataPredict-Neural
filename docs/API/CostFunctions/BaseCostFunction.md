# [API Reference](../../API.md) - [CostFunctions](../CostFunctions.md) - BaseCostFunction

BaseCostFunction is a base for all optimizers.

## Constructors

### new()

Creates a new cost function object.

```

BaseCostFunction.new(): CostFunctionObject

```

#### Returns:

* CostFunction: The generated cost function object.

## Functions

### calculateCostValue()

```

BaseCostFunction:calculateCostValue(generatedLabelTensor: tensor, labelTensor: tensor)

```

Parameters:

* generatedLabelTensor: The generated label tensor produced by a model.

* labelTensor: The original label tensor associated with the featureTensor

* numberOfData: The value to divide with cost value.

Returns:

* costValue: The calculated cost value.

### calculateCostValue()

```

BaseCostFunction:calculateLossTensor(generatedLabelTensor: tensor, labelTensor: tensor)

```

Parameters:

* generatedLabelTensor: The generated label tensor produced by a model.

* labelTensor: The original label tensor associated with the data.

* numberOfData: The value to divide with cost value.

Returns:

* lossTensor: The calculated loss tensor.

### setCostFunction()

```

BaseCostFunction:setCostFunction(CostFunction: function)

```

Parameters:

* CostFunction: The cost function to be set.

### setLossFunction()

```

BaseCostFunction:setLossFunction(LossFunction: function)

```

Parameters:

* LossFunction: The loss function to be set.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)
