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

local OperatorBlocks = DataPredictNeural.OperatorBlocks

local Add = require(OperatorBlocks.Add)

local Multiply = require(OperatorBlocks.Multiply)

local Subtract = require(OperatorBlocks.Subtract)

local ActivationBlocks = DataPredictNeural.ActivationBlocks

local Sigmoid = require(ActivationBlocks.Sigmoid)

local Tanh = require(ActivationBlocks.Tanh)

local HolderBlocks = DataPredictNeural.HolderBlocks

local InputHolder = require(HolderBlocks.InputHolder)

local NullaryFunctionHolder = require(HolderBlocks.NullaryFunctionHolder)

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
	
	local InputGateInputLinear = parameterDictionary.InputGateInputLinear
	
	local InputGateHiddenLinear = parameterDictionary.InputGateHiddenLinear
	
	local InputGateBias = parameterDictionary.InputGateBias
	
	local InputGateAdd = parameterDictionary.InputGateAdd
	
	local InputActivation = parameterDictionary.InputActivation
	
	local ForgetGateInputLinear = parameterDictionary.ForgetGateInputLinear
	
	local ForgetGateHiddenLinear = parameterDictionary.ForgetGateHiddenLinear
	
	local ForgetGateBias = parameterDictionary.ForgetGateBias
	
	local ForgetGateAdd = parameterDictionary.ForgetGateAdd
	
	local ForgetActivation = parameterDictionary.ForgetActivation
	
	local OutputGateInputLinear = parameterDictionary.OutputGateInputLinear
	
	local OutputGateHiddenLinear = parameterDictionary.OutputGateHiddenLinear
	
	local OutputGateBias = parameterDictionary.OutputGateBias
	
	local OutputGateAdd = parameterDictionary.OutputGateAdd
	
	local OutputActivation = parameterDictionary.OutputActivation
	
	local CellInputLinear = parameterDictionary.CellInputLinear
	
	local CellHiddenLinear = parameterDictionary.CellHiddenLinear
	
	local CellBias = parameterDictionary.CellBias
	
	local CellAdd = parameterDictionary.CellAdd
	
	local CellActivation = parameterDictionary.CellActivation
	
	local CellStateMultiply1 = parameterDictionary.CellStateMultiply1
	
	local CellStateMultiply2 = parameterDictionary.CellStateMultiply2
	
	local CellStateAdd = parameterDictionary.CellStateAdd
	
	local CellStateActivation = parameterDictionary.CellStateActivation
	
	local OutputMultiply = parameterDictionary.OutputMultiply
	
	local InputGateInputLinearOptimizer = parameterDictionary.InputGateInputLinearOptimizer

	local InputGateHiddenLinearOptimizer = parameterDictionary.InputGateHiddenLinearOptimizer

	local InputGateBiasOptimizer = parameterDictionary.InputGateBiasOptimizer
	
	local ForgetGateInputLinearOptimizer = parameterDictionary.ForgetGateInputLinearOptimizer
	
	local ForgetGateHiddenLinearOptimizer = parameterDictionary.ForgetGateHiddenLinearOptimizer
	
	local ForgetGateBiasOptimizer = parameterDictionary.ForgetGateBiasOptimizer
	
	local OutputGateInputLinearOptimizer = parameterDictionary.OutputGateInputLinearOptimizer
	
	local OutputGateHiddenLinearOptimizer = parameterDictionary.OutputGateHiddenLinearOptimizer
	
	local OutputGateBiasOptimizer = parameterDictionary.OutputGateBiasOptimizer
	
	local CellInputLinearOptimizer = parameterDictionary.CellInputLinearOptimizer
	
	local CellHiddenLinearOptimizer = parameterDictionary.CellHiddenLinearOptimizer
	
	local CellBiasOptimizer = parameterDictionary.CellBiasOptimizer
	
	local Regularizer = parameterDictionary.Regularizer

	local InputRegularizer = parameterDictionary.InputRegularizer or Regularizer

	local HiddenRegularizer = parameterDictionary.HiddenRegularizer or Regularizer

	local BiasRegularizer = parameterDictionary.BiasRegularizer or Regularizer
	
	local InputGateInputLinearRegularizer = parameterDictionary.InputGateInputLinearRegularizer or InputRegularizer
	
	local InputGateHiddenLinearRegularizer = parameterDictionary.InputGateHiddenLinearRegularizer or HiddenRegularizer
	
	local InputGateBiasRegularizer = parameterDictionary.InputGateBiasRegularizer or BiasRegularizer
	
	local ForgetGateInputLinearRegularizer = parameterDictionary.ForgetGateInputLinearRegularizer or InputRegularizer
	
	local ForgetGateHiddenLinearRegularizer = parameterDictionary.ForgetGateHiddenLinearRegularizer or HiddenRegularizer
	
	local ForgetGateBiasRegularizer = parameterDictionary.ForgetGateBiasRegularizer or BiasRegularizer
	
	local OutputGateInputLinearRegularizer = parameterDictionary.OutputGateInputLinearRegularizer or InputRegularizer
	
	local OutputGateHiddenLinearRegularizer = parameterDictionary.OutputGateHiddenLinearRegularizer or HiddenRegularizer
	
	local OutputGateBiasRegularizer = parameterDictionary.OutputGateBiasRegularizer or BiasRegularizer
	
	local CellInputLinearRegularizer = parameterDictionary.CellInputLinearRegularizer or HiddenRegularizer
	
	local CellHiddenLinearRegularizer = parameterDictionary.CellHiddenLinearRegularizer or HiddenRegularizer
	
	local CellBiasRegularizer = parameterDictionary.CellBiasRegularizer or BiasRegularizer
	
	local inputHiddenDimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}

	local hiddenHiddenDimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}

	local biasDimensionSizeArray = {1, hiddenDimensionSize}
	
	if (not InputGateInputLinear) then InputGateInputLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputGateInputLinearOptimizer, Regularizer = InputGateInputLinearRegularizer}) end
	
	if (not InputGateHiddenLinear) then InputGateHiddenLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputGateHiddenLinearOptimizer, Regularizer = InputGateHiddenLinearRegularizer}) end
	
	if (not InputGateBias) then InputGateBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputGateBiasOptimizer, Regularizer = InputGateBiasRegularizer}) end
	
	if (not InputGateAdd) then InputGateAdd = Add.new() end
	
	if (not InputActivation) then InputActivation = Sigmoid.new() end
	
	if (not ForgetGateInputLinear) then ForgetGateInputLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = ForgetGateInputLinearOptimizer, Regularizer = ForgetGateInputLinearRegularizer}) end
	
	if (not ForgetGateHiddenLinear) then ForgetGateHiddenLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = ForgetGateHiddenLinearOptimizer, Regularizer = ForgetGateHiddenLinearRegularizer}) end
	
	if (not ForgetGateBias) then ForgetGateBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = ForgetGateBiasOptimizer, Regularizer = ForgetGateBiasRegularizer}) end
	
	if (not ForgetGateAdd) then ForgetGateAdd = Add.new() end
	
	if (not ForgetActivation) then ForgetActivation = Sigmoid.new() end
	
	if (not OutputGateInputLinear) then OutputGateInputLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = OutputGateInputLinearOptimizer, Regularizer = OutputGateInputLinearRegularizer}) end
	
	if (not OutputGateHiddenLinear) then OutputGateHiddenLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = OutputGateHiddenLinearOptimizer, Regularizer = OutputGateHiddenLinearRegularizer}) end
	
	if (not OutputGateBias) then OutputGateBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = OutputGateBiasOptimizer, Regularizer = OutputGateBiasRegularizer}) end
	
	if (not OutputGateAdd) then OutputGateAdd = Add.new() end
	
	if (not OutputActivation) then OutputActivation = Sigmoid.new() end
	
	if (not CellInputLinear) then CellInputLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = CellInputLinearOptimizer, Regularizer = CellInputLinearRegularizer}) end
	
	if (not CellHiddenLinear) then CellHiddenLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = CellHiddenLinearOptimizer, Regularizer = CellInputLinearRegularizer}) end
	
	if (not CellBias) then CellBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = CellBiasOptimizer, Regularizer = CellBiasRegularizer}) end
	
	if (not CellAdd) then CellAdd = Add.new() end
	
	if (not CellActivation) then CellActivation = Tanh.new() end
	
	if (not CellStateMultiply1) then CellStateMultiply1 = Multiply.new() end
	
	if (not CellStateMultiply2) then CellStateMultiply2 = Multiply.new() end
	
	if (not CellStateAdd) then CellStateAdd = Add.new() end
	
	if (not CellStateActivation) then CellStateActivation = Tanh.new() end
	
	if (not OutputMultiply) then OutputMultiply = Multiply.new() end
	
	InputGateInputLinear:linkForward(InputGateAdd)
	
	InputGateHiddenLinear:linkForward(InputGateAdd)
	
	InputGateBias:linkForward(InputGateAdd)
	
	InputGateAdd:linkForward(InputActivation)
	
	ForgetGateInputLinear:linkForward(ForgetGateAdd)
	
	ForgetGateHiddenLinear:linkForward(ForgetGateAdd)
	
	ForgetGateBias:linkForward(ForgetGateAdd)
	
	ForgetGateAdd:linkForward(ForgetActivation)
	
	OutputGateInputLinear:linkForward(OutputGateAdd)
	
	OutputGateHiddenLinear:linkForward(OutputGateAdd)
	
	OutputGateBias:linkForward(OutputGateAdd)
	
	OutputGateAdd:linkForward(OutputActivation)
	
	CellInputLinear:linkForward(CellAdd)
	
	CellHiddenLinear:linkForward(CellAdd)
	
	CellBias:linkForward(CellAdd)
	
	CellAdd:linkForward(CellActivation)
	
	ForgetActivation:linkForward(CellStateMultiply1)
	
	InputActivation:linkForward(CellStateMultiply2)
	
	CellActivation:linkForward(CellStateMultiply2)
	
	CellStateMultiply1:linkForward(CellStateAdd)
	
	CellStateMultiply2:linkForward(CellStateAdd)
	
	CellStateAdd:linkForward(CellStateActivation)
	
	CellStateActivation:linkForward(OutputMultiply)
	
	OutputActivation:linkForward(OutputMultiply)
	
	NewSequentialContainer.InputGateInputLinear = InputGateInputLinear
	
	NewSequentialContainer.InputGateHiddenLinear = InputGateHiddenLinear
	
	NewSequentialContainer.InputGateBias = InputGateBias
	
	NewSequentialContainer.InputGateAdd = InputGateAdd
	
	NewSequentialContainer.InputActivation = InputActivation
	
	NewSequentialContainer.ForgetGateInputLinear = ForgetGateInputLinear
	
	NewSequentialContainer.ForgetGateHiddenLinear = ForgetGateHiddenLinear
	
	NewSequentialContainer.ForgetGateBias = ForgetGateBias
	
	NewSequentialContainer.ForgetGateAdd = ForgetGateAdd
	
	NewSequentialContainer.ForgetActivation = ForgetActivation
	
	NewSequentialContainer.OutputGateInputLinear = OutputGateInputLinear
	
	NewSequentialContainer.OutputGateHiddenLinear = OutputGateHiddenLinear
	
	NewSequentialContainer.OutputGateBias = OutputGateBias
	
	NewSequentialContainer.OutputGateAdd = OutputGateAdd
	
	NewSequentialContainer.OutputActivation = OutputActivation
	
	NewSequentialContainer.CellInputLinear = CellInputLinear
	
	NewSequentialContainer.CellHiddenLinear = CellHiddenLinear
	
	NewSequentialContainer.CellBias = CellBias
	
	NewSequentialContainer.CellAdd = CellAdd
	
	NewSequentialContainer.CellActivation = CellActivation
	
	NewSequentialContainer.CellStateMultiply1 = CellStateMultiply1
	
	NewSequentialContainer.CellStateMultiply2 = CellStateMultiply2
	
	NewSequentialContainer.CellStateAdd = CellStateAdd
	
	NewSequentialContainer.CellStateActivation = CellStateActivation
	
	NewSequentialContainer.OutputMultiply = OutputMultiply
	
	NewSequentialContainer.WeightBlockArray = {InputGateInputLinear, InputGateHiddenLinear, InputGateBias, ForgetGateInputLinear, ForgetGateHiddenLinear, ForgetGateBias, OutputGateInputLinear, OutputGateHiddenLinear, OutputGateBias, CellInputLinear, CellHiddenLinear, CellBias}
	
	NewSequentialContainer:setForwardPropagateFunction(function(featureTensor, hiddenStateTensor, cellStateTensor)
		
		local OutputMultiply = NewSequentialContainer.OutputMultiply

		OutputMultiply:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewSequentialContainer.InputActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewSequentialContainer.ForgetActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewSequentialContainer.OutputActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewSequentialContainer.CellActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewSequentialContainer.CellStateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		NewSequentialContainer.InputGateInputLinear:transform(featureTensor)

		NewSequentialContainer.InputGateHiddenLinear:transform(hiddenStateTensor)
		
		NewSequentialContainer.ForgetGateInputLinear:transform(featureTensor)
		
		NewSequentialContainer.ForgetGateHiddenLinear:transform(hiddenStateTensor)
		
		NewSequentialContainer.OutputGateInputLinear:transform(featureTensor)
		
		NewSequentialContainer.OutputGateHiddenLinear:transform(hiddenStateTensor)
		
		NewSequentialContainer.CellInputLinear:transform(featureTensor)
		
		NewSequentialContainer.CellHiddenLinear:transform(hiddenStateTensor)
		
		NewSequentialContainer.CellStateMultiply1:transform(cellStateTensor)

		local transformedTensor = OutputMultiply:waitForTransformedTensor()

		return transformedTensor
		
	end)
	
	NewSequentialContainer:setBackwardPropagateFunction(function(lossTensor) 
		
		NewSequentialContainer.OutputMultiply:differentiate(lossTensor)

		local inputGateInputLinearLossTensor = NewSequentialContainer.InputGateInputLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local inputGateHiddenLinearLossTensor = NewSequentialContainer.InputGateHiddenLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local inputGateBiasLossTensor = NewSequentialContainer.InputGateBias:waitForTotalFirstDerivativeTensorArray()[1]
		
		local forgetGateInputLinearLossTensor = NewSequentialContainer.ForgetGateInputLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local forgetGateHiddenLinearLossTensor = NewSequentialContainer.ForgetGateHiddenLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local forgetGateBiasLossTensor = NewSequentialContainer.ForgetGateBias:waitForTotalFirstDerivativeTensorArray()[1]
		
		local outputGateInputLinearLossTensor = NewSequentialContainer.OutputGateInputLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local outputGateHiddenLinearLossTensor = NewSequentialContainer.OutputGateHiddenLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local outputGateBiasLossTensor = NewSequentialContainer.OutputGateBias:waitForTotalFirstDerivativeTensorArray()[1]
		
		local cellInputLinearLossTensor = NewSequentialContainer.CellInputLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local cellHiddenLinearLossTensor = NewSequentialContainer.CellHiddenLinear:waitForTotalFirstDerivativeTensorArray()[1]
		
		local cellBiasLossTensor = NewSequentialContainer.CellBias:waitForTotalFirstDerivativeTensorArray()[1]

		local weightTensorArray = {inputGateInputLinearLossTensor, inputGateHiddenLinearLossTensor, inputGateBiasLossTensor, forgetGateInputLinearLossTensor, forgetGateHiddenLinearLossTensor, forgetGateBiasLossTensor, outputGateInputLinearLossTensor, outputGateHiddenLinearLossTensor, outputGateBiasLossTensor, cellInputLinearLossTensor, cellHiddenLinearLossTensor, cellBiasLossTensor} 

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
