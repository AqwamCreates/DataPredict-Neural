# [API Reference](../../API.md) - [CostFunctions](../CostFunctions.md) - FocalLoss

## Constructors

### new()

Creates a new cost function object.

```

FocalLoss.new({alpha: number, gamma: number}): CostFunctionObject

```

### Parameters

* alpha: The weighting factor used to deal with class imbalance. Must be between 0 and 1.

* gamma: A tunable focusing parameter. Must be a positive value.

## Functions

### setParameters()

Creates a new cost function object.

```

FocalLoss:setParameters({alpha: number, gamma: number})

```

### Parameters

* alpha: The weighting factor used to deal with class imbalance. Must be between 0 and 1.

* gamma: A tunable focusing parameter. Must be a positive value.

## Inherited From

* [BaseCostFunction](BaseCostFunction.md)

## References

* [Focal loss implementation for LightGBM](https://maxhalford.github.io/blog/lightgbm-focal-loss/#:~:text=The%20following%20plot%20shows%20the,is%20convex%20in%20every%20case.)
