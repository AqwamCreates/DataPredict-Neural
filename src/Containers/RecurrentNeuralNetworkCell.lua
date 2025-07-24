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
	
	if ((type(inputDimensionSize) ~= "number") and (not InputLinear)) then error("Invalid input dimension size") end
	
	if ((type(hiddenDimensionSize) ~= "number") and (not (InputLinear and InputBias and HiddenLinear and HiddenBias))) then error("Invalid hidden dimension size") end
	
	if (not InputLinear) then InputLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not InputBias) then InputBias = require(DataPredictNeural.WeightBlocks.Bias).new({dimensionSizeArray = {1, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not HiddenLinear) then HiddenLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not HiddenBias) then HiddenBias = require(DataPredictNeural.WeightBlocks.Bias).new({dimensionSizeArray = {1, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
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

	self.cutOffValue = cutOffValue or self.cutOffValue

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