# Creating Our First Neural Network

In this tutorial, we will show you on how to create the most basic neural network. We will split this tutorial into multiple smaller sections so that you do not get overwhelmed with too much information.

Here are the list of section for you to warm up before reading through them.

* Gentle introduction to function blocks
	
* Setting up the weights
	
* Choosing a cost function
	
* Building the model
	
## Gentle Introduction To Function Blocks

Function blocks are the building blocks for neural networks. These blocks handles the following tasks:

* Transforming inputs to certain outputs.
	
* Calculating first-order derivatives for a given value.
	
* Storing the inputs, transformed inputs and first-order derivative values.
	
Function blocks are typically inherited by:

* Activation blocks: Converts values into another.
	
* Weight blocks: Holds the weights for our neural network.
	
* Shape transformation blocks: Change the shape of our tensors without modifying the values.

* Any many others!

## Setting Up The Weights

In order for our neural network to learn, we need to store the weight values for our neural network. These are typically stored in the weight blocks. In order to create them, we first need to set the dimension sizes of the weights. 

Below, it shows a code creating a layer block containing a weight tensor of specific size. It is also setting up learning rate value when the neural network performs a gradient descent.

```lua

local WeightBlocks = DataPredictNeural.WeightBlocks

local Linear = WeightBlocks.Linear.new({

	dimensionArray = {1, 90, 4},
	
	learningRate = 0.01

})

```

As you can see, the length of dimension array is equal to total number of dimensions, and the values inside it represent the sizes of each dimension.

Currently, the library can only handle 3 dimensional tensors only. But this might change in the future.

## Choosing A Cost Function

The cost function tells us the performance of the model making predictions compared to the label values. When the predictions are close to the label values, the cost is low. Otherwise, the cost is high. By minimizing this cost, the network gets better at capturing the patterns in our data and making more accurate predictions.

Below, we will show a commented code of two important functions related to cost function.

```lua

local CostFunction = DataPredictNeural.CostFunctions.MeanSquaredError.new()

local lossTensor = CostFunction:calculateLossTensor(generatedLabelTensor, labelTensor) -- The calculateLossTensor() function is used to calculate the difference between the two tensors.
	
local costValue = CostFunction:calculateCostValue(generatedLabelTensor, labelTensor) -- The calculateCostValue() function is used to calculate the overall cost or error between the output and label tensors.

```

With the fundamental knowledge in place, we can now create our first neural network.

## Building The Model

Below, we will show you a block of code and describe what each line of code are doing through the comments.

```lua

local DataPredictNeural = require(DataPredictNeural)

local TensorL3D = require(TensorL3D)

local SequentialNeuralNetwork = DataPredictNeural.Containers.Sequential.new() -- For this tutorial, we want to create a basic neural network. So, we will use a "Sequential" container that holds all of our blocks and also to automatically set up necessary connections between the blocks.

local WeightBlocks = DataPredictNeural.WeightBlocks

local ActivationBlocks = DataPredictNeural.ActivationBlocks

local TransformationBlocks = DataPredictNeural.TransformationBlocks

local CostFunction = DataPredictNeural.CostFunctions.MeanSquaredError.new()

local inputTensor = TensorL3D:createRandomUniformTensor({1, 4, 1}) -- Generating our input tensor here. Pay attention to the dimensions.

local labelTensor = TensorL3D:createRandomNormalTensor({1, 4, 3}) -- Generating our label tensor here. Pay attention to the dimensions here as well.

--[[

In order for us to be able to calculate the loss tensor, we need to make sure the generated label tensor dimensions matches with the original one.

When initializing the weights, ensure that the 3rd dimension of the input tensor matches the 2nd dimension of the weight tensor. In addition, ensure that both have the same sizes for the first dimension.

When doing the dot product between the input tensor and weight tensor, it will give a new tensor shape.

	* Input tensor: {a, b, c}
	
	* Weight tensor: {a, c, d}
	
	* Output tensor: {a, b, d}

Below, we will demonstrate how the tensor shape changes as we add blocks to our "Sequential" container.

--]]

SequentialNeuralNetwork:setMultipleFunctionBlocks( -- Input tensor starts with the size of {1, 4, 1}.
	
	WeightBlocks.Linear.new({dimensionSizeArray = {1, 1, 3}}), -- {1, 4, 1} * {1, 1, 3} -> {1, 4, 3}
	
	WeightBlocks.Linear.new({dimensionSizeArray = {1, 3, 5}}), -- {1, 4, 3} * {1, 3, 5} -> {1, 4, 5}
	
	ActivationBlocks.LeakyReLU.new(),
	
	WeightBlocks.Linear.new({dimensionSizeArray = {1, 5, 1}}), -- {1, 4, 5} * {1, 5, 3} -> {1, 4, 3}
	
	ActivationBlocks.LeakyReLU.new()
	
)

--[[

Since we now have verified that the generated label tensor has the same shape as the label tensor, we can now perform the training.

--]]

for i = 1, 100000 do
	
	local generatedLabelTensor = SequentialNeuralNetwork:forwardPropagate(inputTensor) -- Generate our label tensor first.

	local lossTensor = CostFunction:calculateLossTensor(generatedLabelTensor, labelTensor) -- Calculate the loss tensor for backpropagation.
	
	local costValue = CostFunction:calculateCostValue(generatedLabelTensor, labelTensor)
	
	SequentialNeuralNetwork:backPropagate(lossTensor) -- Pass the loss tensor to backpropagate() function to update the weights.
	
	print(costValue)
	
	task.wait()
	
end

```

Additionally, we can also do a backpropagation with the code below, but this allows us to do distributed training.

```lua

for i = 1, 100000 do
	
	local generatedLabelTensor = SequentialNeuralNetwork:forwardPropagate(inputTensor) -- Generate our label tensor first.

	local lossTensor = CostFunction:calculateLossTensor(generatedLabelTensor, labelTensor) -- Calculate the loss tensor for backpropagation.
	
	local costValue = CostFunction:calculateCostValue(generatedLabelTensor, labelTensor)
	
	local weightLossTensorArray = SequentialNeuralNetwork:calculateWeightLossTensorArray(lossTensor) -- Calculate the weight loss tensors for our weights. This table can be sent to another neural network of the same architecture if you want to do distributed training.
	
	SequentialNeuralNetwork:gradientDescent(weightLossTensorArray) -- Pass the weight loss array to the gradientDescent() function to update the weights.
	
	print(costValue)
	
	task.wait()
	
end

```

Whew! That's quite a lot to take in, wasn't it?

That being said, you are now equipped with the fundamental knowledge on how to use this deep learning library.

Now, go have some fun with it!
