# Getting Started With Deep Reinforcement Learning

## Requirements

* Knowledge on how to build neural networks, which can be found [here](CreatingOurFirstNeuralNetwork.md).

## What Is Reinforcement Learning?

Reinforcement learning is a way for our models to learn on its own without the labels.

We can expect our models to perform poorly at the start of the training but they will gradually improve over time.

## The Basics

### Environment Feature Tensor

An environment feature tensor is a tensor containing all the information related to model's environment. It can contain as many information such as:

* Distance

* Health

* Speed

An example of environment feature tensor will look like this:

```lua
local environmentFeatureTensor = {

  {1, -32, 234, 12, -97} -- 1 is added at first column for bias, but it is optional.

}
```

### Reward Value

This is the value where we reward or punish the models. The properties of reward value is shown below:

* Positive value: Reward

* Negative Value: Punishment

* Large value: Large reward / punishment

* Small value: Small reward / punishment

It is recommended to set the reward that is within the range of:

```lua
-1 <= (total reward * learning rate) <= 1
```

### Action Labels

Action label is a label produced by the model. This label can be a part of decision-making classes or classification classes. For example:

* Decision-making classes: "Up", "Down", "Left", "Right", "Forward", "Backward"

* Classification classes: 1, 2, 3, 4, 5, 6

## Setting Up Our Reinforcement Learning Model

There are two ways we can setup our models for our environment:

* Classic Setup (Recommended for experts or people who wants more control over their models)

* Quick Setup (Recommended for beginners)

Below we will show you the difference between the two above. But first, let's define a number of variables and setup the first part of the model.

```lua

local ClassesList = {1, 2}

local NeuralNetwork = DataPredictNeural.Container.Sequential.new() -- Create the NeuralNetwork first.

NeuralNetwork:setMultipleFunctionBlocks(

  DataPredictNeural.WeightBlocks.Linear.new({dimensionSizeArray = {4, 2}}),

  DataPredictNeural.WeightBlocks.Bias.new({dimensionSizeArray = {1, 2}}),

  DataPredictNeural.ActivationFunctions.LeakyReLU.new()

)

NeuralNetwork:setClassesList(ClassesList)

local DeepQLearning = DataPredict.Models.DeepQLearning.new() -- Then create the DeepQLearning.

DeepQLearning:setModel(NeuralNetwork) -- Then put the NeuralNetwork inside DeepQLearning.

```

## Classic Setup

All the reinforcement learning models have two important functions: 

* update()

  * categoricalUpdate() is for discrete action spaces

  * diagonalGaussianUpdate() is for continuous action spaces

* episodeUpdate()

Below, I will show a code sample using these functions.

```lua

while true do

  local previousEnvironmentFeatureTensor = {{0, 0, 0, 0, 0}} -- We must keep track our previous feature tensor.

  local action = 1

  for step = 1, 1000, 1 do

    local currentEnvironmentFeatureTensor = fetchEnvironmentFeatureTensor(previousEnvironmentFeatureTensor, action)

    action = DeepQLearning:predict(currentEnvironmentFeatureTensor)[1][1]

    local reward = getReward(currentEnvironmentFeatureTensor)

    DeepQLearning:categoricalUpdate(previousEnvironmentFeatureTensor, reward, action, currentEnvironmentFeatureTensor, 0) -- update() is called whenever a step is made. The value of zero indicates that the current environment feature tensor is not a terminal state.

    previousEnvironmentFeatureTensor = currentEnvironmentFeatureTensor

    local hasGameEnded = checkIfGameHasEnded(environmentTensor)

    if hasGameEnded then break end

  end

  QLearningNeuralNetwork:episodeUpdate(1) -- episodeUpdate() is used whenever an episode ends. An episode is the total number of steps that determines when the model should stop training. The value of one indicates that the current environment feature tensor is a terminal state.

end

```

As you can see, there are a lot of things that we must track of, but it gives you total freedom on what you want to do with the reinforcement learning models.

## Quick Setup

To reduce the amount of things we need to track, we can use CategoricalPolicy in "QuickSetups" section. 

Note that you need to use the code from the [DataPredict](https://aqwamcreates.github.io/DataPredict/) library and must not be confused with DataPredict Neural library.

```lua

local DeepQLearningQuickSetup = DataPredict.QuickSetups.CategoricalPolicy.new()

DeepQLearningQuickSetup:setModel(DeepQLearning)

DeepQLearningQuickSetup:setClassesList(ClassesList)

local environmentFeatureTensor = {{0, 0, 0, 0, 0}}

local action = 1

while true do

  environmentFeatureTensor = fetchEnvironmentFeatureTensor(environmentFeatureTensor, action)

  local reward = getReward(environmentFeatureTensor, action)

  action = DeepQLearningQuickSetup:reinforce(environmentFeatureTensor, reward)

end

```

As you can see, the CategoricalPolicy compresses a number of codes into reinforce() function.

# Wrapping It All Up

In this tutorial, you have learnt the starting point of the reinforcement learning neural networks. 

These only cover the basics. You can find more information here:

* [Deep Q-Learning](../API/Models/DeepQLearning.md)

* [Deep SARSA](../API/Models/DeepStateActionRewardStateAction.md)

* [Deep Expected SARSA](../API/Models/DeepExpectedStateActionRewardStateAction.md)
