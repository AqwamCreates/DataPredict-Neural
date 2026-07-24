--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Neural)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local DataPredictNeural = script.Parent.Parent

local AqwamTensorLibrary = require(DataPredictNeural.AqwamTensorLibraryLinker.Value)

local BaseContainer = require(script.Parent.BaseContainer)

local WeightBlocks = DataPredictNeural.WeightBlocks

local Linear = require(WeightBlocks.Linear)

local Bias = require(WeightBlocks.Bias)

local RecurrentNeuralNetworkCellContainer = {}

RecurrentNeuralNetworkCellContainer.__index = RecurrentNeuralNetworkCellContainer

setmetatable(RecurrentNeuralNetworkCellContainer, BaseContainer)

local defaultCutOffValue = 0

function RecurrentNeuralNetworkCellContainer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecurrentNeuralNetworkCellContainer = BaseContainer.new(parameterDictionary)

	setmetatable(NewRecurrentNeuralNetworkCellContainer, RecurrentNeuralNetworkCellContainer)

	NewRecurrentNeuralNetworkCellContainer:setName("RecurrentNeuralNetworkCell")
	
	local inputDimensionSize = parameterDictionary.inputDimensionSize
	
	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize
	
	local learningRate = parameterDictionary.learningRate
	
	local weightInitializationMode = parameterDictionary.weightInitializationMode
	
	local InputLinear = parameterDictionary.InputLinear
	
	local InputBias = parameterDictionary.InputBias
	
	local HiddenLinear = parameterDictionary.HiddenLinear
	
	local HiddenBias = parameterDictionary.HiddenBias
	
	local Add = parameterDictionary.Add
	
	local Activation = parameterDictionary.Activation
	
	local InputLinearOptimizer = parameterDictionary.InputLinearOptimizer
	
	local InputBiasOptimizer = parameterDictionary.InputBiasOptimizer
	
	local HiddenLinearOptimizer = parameterDictionary.HiddenLinearOptimizer
	
	local HiddenBiasOptimizer = parameterDictionary.HiddenBiasOptimizer
	
	local InputLinearRegularizer = parameterDictionary.InputLinearRegularizer
	
	local InputBiasRegularizer = parameterDictionary.InputBiasRegularizer
	
	local HiddenLinearRegularizer = parameterDictionary.HiddenLinearRegularizer
	
	local HiddenBiasRegularizer = parameterDictionary.HiddenBiasRegularizer
	
	local inputHiddenDimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}

	local hiddenHiddenDimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}

	local biasDimensionSizeArray = {1, hiddenDimensionSize}
	
	if (not InputLinear) then InputLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputLinearOptimizer, Regularizer = InputLinearRegularizer}) end
	
	if (not InputBias) then InputBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputBiasOptimizer, Regularizer = InputBiasRegularizer}) end
	
	if (not HiddenLinear) then HiddenLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = HiddenLinearOptimizer, Regularizer = HiddenLinearRegularizer}) end
	
	if (not HiddenBias) then HiddenBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = HiddenBiasOptimizer, Regularizer = HiddenBiasRegularizer}) end
	
	if (not Add) then Add = require(DataPredictNeural.OperatorBlocks.Add).new() end
	
	if (not Activation) then Activation = require(DataPredictNeural.ActivationBlocks.Tanh).new() end
	
	InputLinear:linkForward(InputBias)
	
	HiddenLinear:linkForward(HiddenBias)
	
	InputBias:linkForward(Add)
	
	HiddenBias:linkForward(Add)
	
	Add:linkForward(Activation)
	
	NewRecurrentNeuralNetworkCellContainer.inputDimensionSize = inputDimensionSize
	
	NewRecurrentNeuralNetworkCellContainer.hiddenDimensionSize = hiddenDimensionSize
	
	NewRecurrentNeuralNetworkCellContainer.InputLinear = InputLinear
	
	NewRecurrentNeuralNetworkCellContainer.InputBias = InputBias
	
	NewRecurrentNeuralNetworkCellContainer.HiddenLinear = HiddenLinear

	NewRecurrentNeuralNetworkCellContainer.HiddenBias = HiddenBias
	
	NewRecurrentNeuralNetworkCellContainer.Add = Add
	
	NewRecurrentNeuralNetworkCellContainer.Activation = Activation
	
	NewRecurrentNeuralNetworkCellContainer.WeightBlockArray = {InputLinear, InputBias, HiddenLinear, HiddenBias}
	
	NewRecurrentNeuralNetworkCellContainer.OutputBlockArray = {Activation}
	
	NewRecurrentNeuralNetworkCellContainer.ClassesList = parameterDictionary.ClassesList or {}

	NewRecurrentNeuralNetworkCellContainer.cutOffValue = parameterDictionary.cutOffValue or defaultCutOffValue
	
	NewRecurrentNeuralNetworkCellContainer:setForwardPropagateFunction(function(featureTensor, hiddenStateTensor)
		
		local Activation = NewRecurrentNeuralNetworkCellContainer.Activation

		Activation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		NewRecurrentNeuralNetworkCellContainer.InputLinear:transform(featureTensor)

		NewRecurrentNeuralNetworkCellContainer.HiddenLinear:transform(hiddenStateTensor)

		local transformedTensor = Activation:waitForTransformedTensor()

		return transformedTensor
		
	end)
	
	NewRecurrentNeuralNetworkCellContainer:setConvertToClassTensorFunction(function(transformedTensorArray)

		return NewRecurrentNeuralNetworkCellContainer:convertToClassTensor(transformedTensorArray[1], NewRecurrentNeuralNetworkCellContainer.ClassesList, NewRecurrentNeuralNetworkCellContainer.cutOffValue)

	end)

	return NewRecurrentNeuralNetworkCellContainer

end

function RecurrentNeuralNetworkCellContainer:setCutOffValue(cutOffValue)

	self.cutOffValue = cutOffValue

end

function RecurrentNeuralNetworkCellContainer:getCutOffValue()

	return self.cutOffValue

end

function RecurrentNeuralNetworkCellContainer:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function RecurrentNeuralNetworkCellContainer:getClassesList()

	return self.ClassesList

end

return RecurrentNeuralNetworkCellContainer
