# Understanding Function Blocks, Dynamic Computational Graphs And Automatic-Manual Differentiation

## Function Blocks

Functional blocks are the very core of this library. They enable us to do automatic differentiation so that we don't have to manually carry out the partial derivatives of any functions.

These two functions are the most important parts of the function blocks:

* transform()

* differentiate()
	
Later, we will explore how these functions work. But first, we need to create the function block before we get started. The code is shown below:

```lua

local FunctionBlock = DataPredictNeural.Cores.FunctionBlock.new()

```

### The Transform Function

The transform() function converts the input to certain outputs. This can be achieved by writing:

```lua

local transformedInputTensor = FunctionBlock:transform(inputTensor)

```

Typically, the activation, weight and transformation blocks saves the input tensor and transformed input tensor for calculating first-order derivatives. But these can be manually changed using setSaveInputTensor() and setSaveTransformedInputTensor() functions.

### The Differentiate Function

The differentiate() function uses a given value to find the first-order derivative values for that particular function. This can be achieved by writing:

```lua

local firstDerivativeTensor = FunctionBlock:differentiate(tensorToBeDifferentiated)

```

Just like the transform() function, the first-order derivative values are typically stored in function blocks. This can be manually changed using setSaveFirstDerivativeTensor() function.

Additionally, you can put your own "transformedInputTensor" and "inputTensor" to the function's arguments.

```lua

local firstDerivativeTensor = FunctionBlock:differentiate(tensorToBeDifferentiated, transformedInputTensor, inputTensor)

```

When "tensorToBeDifferentiated" is not given, it will use a seed. The seed is always a tensor containing values of 1 and always has the same tensor shape to "transformedInputTensor".

Since we have covered the basics of function blocks, we can now look into dynamic computational graphs.

## Dynamic Computational Graphs

From our previous tutorial, you have seen that the function blocks are arranged in sequential order when contained inside "Sequential" container. This is a very basic neural network architecture.

But what if I tell you that the model can be more complex than that? Or rather create any designs that are limited by your own imagination?

This is where dynamic computational graphs comes in.

### Function Block Chaining

In order to create complex models, our function blocks are equipped with two powerful functions: addNextFunctionBlock() and addPreviousFunctionBlock(). These functions allows the function blocks to communicate with the blocks that are in front and behind them.

You can add as many function blocks you want. A single function block can have multiple "next" function blocks or multiple "previous" function blocks.

I will show you an example below. The code is not meant to be run and was written for simplicity.

```lua

local FunctionBlock = DataPredictNeural.Cores.FunctionBlock

-- Setting up our "main" function block.

local MainFunctionBlock = FunctionBlock.new()

-- Then we set up three "next" function blocks.

local NextFunctionBlock1 = FunctionBlock.new()

local NextFunctionBlock2 = FunctionBlock.new()

local NextFunctionBlock3 = FunctionBlock.new()

-- Then we chain the three "next" function blocks to the "main" function block.

MainFunctionBlock:addNextFunctionBlock(NextFunctionBlock1)

MainFunctionBlock:addNextFunctionBlock(NextFunctionBlock2)

MainFunctionBlock:addNextFunctionBlock(NextFunctionBlock3)

-- Lastly we chain the "next" function blocks back to the "main" function block.

NextFunctionBlock1:addPreviousFunctionBlock(MainFunctionBlock)

NextFunctionBlock2:addPreviousFunctionBlock(MainFunctionBlock)

NextFunctionBlock3:addPreviousFunctionBlock(MainFunctionBlock)

```

Now you have created a complex neural network model using computational graph.

And it gets even better.

### The Transform And Differentiate Function Chaining

When you call transform() at the "main" block, the transformedInputTensor will be passed to "next" function blocks.

Below, I will show you how you retrieve the final result after the inputTensor being passed to multiple function blocks. Again, the code is not meant to be run and was written for simplicity.

```lua

MainFunctionBlock:transform(inputTensor) -- First, lets put in an inputTensor.

local transformedInputTensor1 = NextFunctionBlock1:getTransformedInputTensor() -- This is the first way to get the final result.

local transformedInputTensor2 = NextFunctionBlock2:waitForTransformedInputTensor() -- You can also wait for it to be available if you expect the calculation time to be long.

```

In the "main" function block transform() function, it calls the transform() function from the "next" function blocks. Each of the "next" function blocks will then run in separate threads, making it suitable for model parallelism.

For the differentiate() function, the process is the same, but in reverse.

```lua

NextFunctionBlock1:differentiate(tensorToBeDifferentiated1) -- Let's differentiate three different tensors.

NextFunctionBlock2:differentiate(tensorToBeDifferentiated2)

NextFunctionBlock3:differentiate(tensorToBeDifferentiated3)

local firstDerivativeTensor = MainFunctionBlock:waitForFirstDerivativeTensor() -- Wait for the first derivative tensor.

```

You may need to create new block classes that inherits from FunctionBlock if you wish to combine different tensors upon receiving them.

Or you can do something like this:

```lua

local firstDerivativeTensor1 = NextFunctionBlock1:differentiate(tensorToBeDifferentiated1)

local firstDerivativeTensor2 = NextFunctionBlock2:differentiate(tensorToBeDifferentiated2)

local firstDerivativeTensor3 = NextFunctionBlock3:differentiate(tensorToBeDifferentiated3)

local combinedFirstDerivativeTensor = firstDerivativeTensor1 + firstDerivativeTensor2 + firstDerivativeTensor3

local mainFirstDerivativeTensor = MainFunctionBlock:differentiate(combinedFirstDerivativeTensor)

```

As you can see, computational graphs are very powerful and allows us to build any kind of neural network models that we want.

## Automatic-Manual Differentiation

The function blocks uses reverse-mode automatic differentiation where it collects the partial derivative from the last function block to the first function block.

Also, you may have noticed that we sometimes need to figure out some of the first derivative function. This is because the functional blocks does not provide full automatic differentiation and instead combines both automatic and manual differentiation.

The reason for this design was due to the fact full automatic differentiation is computationally expensive. So, a hybrid approach was chosen so that the calculations are more performant while making sure the programmers don't have to deal with the majority of the math calculations.

And since you're here, you now have learnt the majority of the knowledge that this deep learning library has to offer. Congratulations!
