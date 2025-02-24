# Saving And Loading Model Weights

DataPredict Neural provides the ability to save and load weights from trained models. There are two ways to access these weights.

* Function Blocks That Are Part Of BaseWeightBlock class.

* Containers.

## Saving And Loading Weights From The BaseWeightBlock Classes

In order to save the weights from BaseWeightBlock classes, we first need to call the getWeightTensor() function on one of our weight blocks.

```lua

local savedWeightTensor = Linear:getWeightTensor()

```

This should make a deep copy of the weights to savedWeightTensor variable.

To load a weightTensor, all you need to do is to call the setWeightTensor() function.

```lua

Linear:setWeightTensor(savedWeightTensor)

```

## Saving And Loading Weights From The Containers

In order to save the weights from Containers, we first need to call the getWeightTensorArray() function.

```lua

local savedWeightTensorArray = Sequential:getWeightTensorArray()

```

This should make a deep copy of the weights to savedWeightTensorArray variable.

To load a weightTensorArray, all you need to do is to call the setWeightTensorArray() function.

```lua

Sequential:setWeightTensorArray(savedWeightTensorArray)

```

## What To Do With The Weights?

You have two ways of saving the weights:

* Storing it to DataStores.

* Copy paste the text printed out by the TensorL library and place it in a text file or Roblox's ModuleScripts.

## Wrapping up

Saving and loading on DataPredict Neural has never been easier. All you need is to call few lines of codes and you're off!

That's all you need to do. Pretty simple, right?

Thank you very much for reading this tutorial!
