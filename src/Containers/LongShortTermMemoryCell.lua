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

local LongShortTermMemoryCellContainer = {}

LongShortTermMemoryCellContainer.__index = LongShortTermMemoryCellContainer

setmetatable(LongShortTermMemoryCellContainer, BaseContainer)

function LongShortTermMemoryCellContainer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewSequentialContainer = BaseContainer.new(parameterDictionary)

	setmetatable(NewSequentialContainer, LongShortTermMemoryCellContainer)

	NewSequentialContainer:setName("LongShortTermMemoryCell")
	
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
	
	NewSequentialContainer.InputLinear = InputLinear
	
	NewSequentialContainer.InputBias = InputBias
	
	NewSequentialContainer.HiddenLinear = HiddenLinear

	NewSequentialContainer.HiddenBias = HiddenBias
	
	NewSequentialContainer.Add = Add
	
	NewSequentialContainer.Activation = Activation
	
	NewSequentialContainer.WeightBlockArray = {InputLinear, InputBias, HiddenLinear, HiddenBias}
	
	NewSequentialContainer:setForwardPropagateFunction(function(featureTensor, hiddenStateTensor)
		
		local Activation = NewSequentialContainer.Activation

		Activation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		task.spawn(function() NewSequentialContainer.InputLinear:transform(featureTensor) end)

		task.spawn(function() NewSequentialContainer.HiddenLinear:transform(hiddenStateTensor) end)

		local transformedTensor = Activation:waitForTransformedTensor()

		return transformedTensor
		
	end)
	
	NewSequentialContainer:setBackwardPropagateFunction(function(lossTensor) 
		
		NewSequentialContainer.Activation:differentiate(lossTensor)

		local inputLinearLossTensor = NewSequentialContainer.InputLinear:waitForTotalFirstDerivativeTensorArray()[1]

		local inputBiasLossTensor = NewSequentialContainer.InputBias:waitForTotalFirstDerivativeTensorArray()[1]

		local hiddenLinearLossTensor = NewSequentialContainer.HiddenLinear:waitForTotalFirstDerivativeTensorArray()[1]

		local hiddenBiasLossTensor = NewSequentialContainer.HiddenBias:waitForTotalFirstDerivativeTensorArray()[1]

		local weightTensorArray = {inputLinearLossTensor, inputBiasLossTensor, hiddenLinearLossTensor, hiddenBiasLossTensor}

		return weightTensorArray
		
	end)

	return NewSequentialContainer

end

function LongShortTermMemoryCellContainer:clearAllStoredTensors()

	for i, WeightBlock in ipairs(self.WeightBlockArray) do 

		WeightBlock:clearAllStoredTensors()

	end
	
	self.Add:clearAllStoredTensors()
	
	self.Activation:clearAllStoredTensors()

end

return LongShortTermMemoryCellContainer