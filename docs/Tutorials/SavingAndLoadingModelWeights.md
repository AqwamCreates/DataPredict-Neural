# Saving And Loading Model Weights

DataPredict Neural provides the ability to save and load weights from trained models. There are two ways to access these weights.

* Function Blocks That Are Part Of BaseWeightBlock class.

* Containers

## Saving And Loading Weights From BaseWeightBlock Classes

In order to save the weights, we first need to call the getWeightTensor() function on one of our weight blocks.

```lua

local savedWeightTensor = Linear:getWeightTensor()

```

This should make a deep copy of the weights to savedWeightTensor variable.

You have two ways of saving the weights:

1. Storing it to DataStores.

2. Copy paste the text printed out by the TensorL library and place it in a text file or Roblox's ModuleScripts

To load a weightTensor, all you need to do is to call the setWeightTensor() function on our weight block.

```lua

Linear:setWeightTensor(savedWeightTensor)

```

Additionally, if you had saved your model as a text file, then you can copy paste the content to a module script and require it to a new variable. Once that is done, you can load the model parameters as shown above

# Wrapping up

Saving and loading on DataPredict Neural has never been easier. All you need is to call few lines of codes and you're off!

That's all you need to do. Pretty simple, right?

Thank you very much for reading this tutorial!
