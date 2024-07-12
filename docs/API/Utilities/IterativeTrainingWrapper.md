# [API Reference](../../API.md) - [Utilities](../Utilities.md) - IterativeTrainingWrapper

## Constructors

### new()

Creates a new iterative training wrapper object. If there are no parameters given for that particular argument, then that argument will use default value (except for Model and CostFunction).

```

IterativeTrainingWrapper.new({maxNumberOfIterations: number, Model: ModelObject, CostFunctionArray: {CostFunctionObject}, targetCostValueUpperBoundArray: {number}, targetCostValueLowerBoundArray: {number}, numberOfIterationsToCheckIfConvergedArray: {number}, numberOfIterationsPerCostCalculation: number, isOutputPrinted: boolean, areUsingArraysAsInputs: boolean, iterationWaitDuration: number/boolean}): IterativeTrainingWrapperObject

```

Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* Model: The model to be used by the iterative training wrapper object.

* CostFunctionArray: An array containing all the cost functions to be used by the iterative training wrapper object.
	
* targetCostValueUpperBoundArray: An array containing all the upper bound of target costs.

* targetCostValueLowerBoundArray: An array containing all the lower bound of target costs.
	
* numberOfIterationsToCheckIfConvergedArray: An array containing all the number of iterations for confirming convergence.
	
* numberOfIterationsPerCostCalculation: The number of iterations for each cost calculation.
	
* isOutputPrinted: A boolean value that specifies if the output is printed.

* areUsingArraysAsInputs: A boolean value that specifies if array of tensor is used.

* iterationWaitDuration: The duration to wait between iterations. Setting it to 'true' will make it wait until the frame is completed.

#### Returns:

IterativeTrainingWrapperObject: The generated iterative training wrapper object.

## Functions

### setParameters()

```

IterativeTrainingWrapper:setParameters({maxNumberOfIterations: number, Model: ModelObject, CostFunctionArray: {CostFunctionObject}, targetCostValueUpperBoundArray: {number}, targetCostValueLowerBoundArray: {number}, numberOfIterationsToCheckIfConvergedArray: {number}, numberOfIterationsPerCostCalculation: number, isOutputPrinted: boolean, areUsingArraysAsInputs: boolean, iterationWaitDuration: number/boolean}): IterativeTrainingWrapperObject

```

Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* Model: The model to be used by the iterative training wrapper object.

* CostFunctionArray: An array containing all the cost functions to be used by the iterative training wrapper object.
	
* targetCostValueUpperBoundArray: An array containing all the upper bound of target costs.

* targetCostValueLowerBoundArray: An array containing all the lower bound of target costs.
	
* numberOfIterationsToCheckIfConvergedArray: An array containing all the number of iterations for confirming convergence.
	
* numberOfIterationsPerCostCalculation: The number of iterations for each cost calculation.
	
* isOutputPrinted: A boolean value that specifies if the output is printed.

* areUsingArraysAsInputs: A boolean value that specifies if array of tensor is used.

* iterationWaitDuration: The duration to wait between iterations. Setting it to 'true' will make it wait until the frame is completed.

### train()

```

IterativeTrainingWrapper:train(featureTensor: tensor/{tensor}, labelTensor: tensor/{tensor}): {{number}}

```

#### Parameters:

* featureTensorArray: An array containing all the feature tensors.

* labelTensorArray: An array containing all the label tensors.

#### Returns:

* costMatrix: A matrix containing the cost values. The each column represents the cost from each output, while each row represents the number of iterations at which the cost values are generated.

## Inherited From:

* [BaseInstance](../Cores/BaseInstance.md)
