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

local GatedRecurrentUnitCellContainer = {}

GatedRecurrentUnitCellContainer.__index = GatedRecurrentUnitCellContainer

setmetatable(GatedRecurrentUnitCellContainer, BaseContainer)

local defaultCutOffValue = 0

function GatedRecurrentUnitCellContainer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewGatedRecurrentUnitCellContainer = BaseContainer.new(parameterDictionary)

	setmetatable(NewGatedRecurrentUnitCellContainer, GatedRecurrentUnitCellContainer)

	NewGatedRecurrentUnitCellContainer:setName("GatedRecurrentUnitCell")
	
	local inputDimensionSize = parameterDictionary.inputDimensionSize
	
	local hiddenDimensionSize = parameterDictionary.hiddenDimensionSize
	
	local learningRate = parameterDictionary.learningRate
	
	local weightInitializationMode = parameterDictionary.weightInitializationMode
	
	local InputResetGateLinear = parameterDictionary.InputResetGateLinear
	
	local HiddenResetGateLinear = parameterDictionary.HiddenResetGateLinear
	
	local ResetGateBias = parameterDictionary.ResetGateBias
	
	local ResetGateAdd = parameterDictionary.ResetGateAdd
	
	local ResetGateActivation = parameterDictionary.ResetGateActivation
	
	local InputUpdateGateLinear = parameterDictionary.InputUpdateGateLinear
	
	local HiddenUpdateGateLinear = parameterDictionary.HiddenUpdateGateLinear
	
	local UpdateGateBias = parameterDictionary.UpdateGateBias
	
	local UpdateGateAdd = parameterDictionary.UpdateGateAdd
	
	local UpdateGateActivation = parameterDictionary.UpdateGateActivation
	
	local InputCandidateLinear = parameterDictionary.InputCandidateLinear

	local HiddenCandidateLinear = parameterDictionary.HiddenCandidateLinear

	local CandidateBias = parameterDictionary.CandidateBias
	
	local CandidateAdd = parameterDictionary.CandidateAdd
	
	local CandidateInputHolder = parameterDictionary.CandidateInputHolder

	local CandidateMultiply = parameterDictionary.CandidateMultiply
	
	local CandidateActivation = parameterDictionary.CandidateActivation
	
	local OutputNullaryFunctionHolder = parameterDictionary.OutputNullaryFunctionHolder
	
	local OutputInputHolder = parameterDictionary.OutputInputHolder
	
	local OutputSubtract = parameterDictionary.OutputSubtract

	local OutputMultiply1 = parameterDictionary.OutputMultiply1
	
	local OutputMultiply2 = parameterDictionary.OutputMultiply2
	
	local OutputAdd = parameterDictionary.OutputAdd
	
	local InputResetGateLinearOptimizer = parameterDictionary.InputResetGateLinearOptimizer
	
	local HiddenResetGateLinearOptimizer = parameterDictionary.HiddenResetGateLinearOptimizer
	
	local ResetGateBiasOptimizer = parameterDictionary.ResetGateBiasOptimizer
	
	local InputUpdateGateLinearOptimizer = parameterDictionary.InputUpdateGateLinearOptimizer
	
	local HiddenUpdateGateLinearOptimizer = parameterDictionary.HiddenUpdateGateLinearOptimizer
	
	local UpdateGateBiasOptimizer = parameterDictionary.UpdateGateBiasOptimizer
	
	local InputCandidateLinearOptimizer = parameterDictionary.InputCandidateLinearOptimizer
	
	local HiddenCandidateLinearOptimizer = parameterDictionary.HiddenCandidateLinearOptimizer
	
	local CandidateBiasOptimizer = parameterDictionary.CandidateBiasOptimizer
	
	local Regularizer = parameterDictionary.Regularizer
	
	local InputRegularizer = parameterDictionary.InputRegularizer or Regularizer

	local HiddenRegularizer = parameterDictionary.HiddenRegularizer or Regularizer

	local BiasRegularizer = parameterDictionary.BiasRegularizer or Regularizer
	
	local InputResetGateRegularizer = parameterDictionary.InputResetGateRegularizer or InputRegularizer
	
	local HiddenResetGateRegularizer = parameterDictionary.HiddenResetGateRegularizer or HiddenRegularizer
	
	local ResetGateBiasRegularizer = parameterDictionary.ResetGateBiasRegularizer or BiasRegularizer
	
	local InputUpdateGateRegularizer = parameterDictionary.InputUpdateGateRegularizer or InputRegularizer
	
	local HiddenUpdateGateRegularizer = parameterDictionary.HiddenUpdateGateRegularizer or HiddenRegularizer
	
	local UpdateGateBiasRegularizer = parameterDictionary.UpdateGateBiasRegularizer or BiasRegularizer
	
	local HiddenCandidateRegularizer = parameterDictionary.HiddenCandidateRegularizer or HiddenRegularizer
	
	local InputCandidateRegularizer = parameterDictionary.InputCandidateRegularizer or InputRegularizer
	
	local CandidateBiasRegularizer = parameterDictionary.CandidateBiasRegularizer or BiasRegularizer
	
	local inputHiddenDimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}
	
	local hiddenHiddenDimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}
	
	local biasDimensionSizeArray = {1, hiddenDimensionSize}
	
	if (not InputResetGateLinear) then InputResetGateLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputResetGateLinearOptimizer, Regularizer = InputResetGateRegularizer}) end
	
	if (not HiddenResetGateLinear) then HiddenResetGateLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = HiddenResetGateLinearOptimizer, Regularizer = HiddenResetGateRegularizer}) end
	
	if (not ResetGateAdd) then ResetGateAdd = Add.new() end
	
	if (not ResetGateBias) then ResetGateBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = ResetGateBiasOptimizer, Regularizer = ResetGateBiasRegularizer}) end
	
	if (not ResetGateActivation) then ResetGateActivation = Sigmoid.new() end
	
	if (not InputUpdateGateLinear) then InputUpdateGateLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputUpdateGateLinearOptimizer, Regularizer = InputUpdateGateRegularizer}) end

	if (not HiddenUpdateGateLinear) then HiddenUpdateGateLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = HiddenUpdateGateLinearOptimizer, Regularizer = HiddenUpdateGateRegularizer}) end
	
	if (not UpdateGateAdd) then UpdateGateAdd = Add.new() end
	
	if (not UpdateGateBias) then UpdateGateBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Regularizer = UpdateGateBiasRegularizer}) end
	
	if (not UpdateGateActivation) then UpdateGateActivation = Sigmoid.new() end
	
	if (not InputCandidateLinear) then InputCandidateLinear = Linear.new({dimensionSizeArray = inputHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = InputCandidateLinearOptimizer, Regularizer = InputCandidateRegularizer}) end

	if (not HiddenCandidateLinear) then HiddenCandidateLinear = Linear.new({dimensionSizeArray = hiddenHiddenDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = HiddenCandidateLinearOptimizer, Regularizer = HiddenCandidateRegularizer}) end
	
	if (not CandidateAdd) then CandidateAdd = Add.new() end

	if (not CandidateBias) then CandidateBias = Bias.new({dimensionSizeArray = biasDimensionSizeArray, learningRate = learningRate, weightInitializationMode = weightInitializationMode, Optimizer = CandidateBiasOptimizer, CandidateBiasRegularizer = CandidateBiasRegularizer}) end
	
	if (not CandidateInputHolder) then CandidateInputHolder = InputHolder.new() end
	
	if (not CandidateMultiply) then CandidateMultiply = Multiply.new() end
	
	if (not CandidateActivation) then CandidateActivation = Tanh.new() end
	
	if (not OutputNullaryFunctionHolder) then OutputNullaryFunctionHolder = NullaryFunctionHolder.new({Function = (function() return 1 end), ChainRuleFirstDerivativeFunction = (function() return end)}) end
	
	if (not OutputInputHolder) then OutputInputHolder = InputHolder.new() end
	
	if (not OutputSubtract) then OutputSubtract = Subtract.new() end
	
	if (not OutputMultiply1) then OutputMultiply1 = Multiply.new() end
	
	if (not OutputMultiply2) then OutputMultiply2 = Multiply.new() end
	
	if (not OutputAdd) then OutputAdd = Add.new() end
	
	InputResetGateLinear:linkForward(ResetGateAdd)
	
	HiddenResetGateLinear:linkForward(ResetGateAdd)
	
	ResetGateAdd:linkForward(ResetGateBias)
	
	ResetGateBias:linkForward(ResetGateActivation)
	
	InputUpdateGateLinear:linkForward(UpdateGateAdd)

	HiddenUpdateGateLinear:linkForward(UpdateGateAdd)

	UpdateGateAdd:linkForward(UpdateGateBias)
	
	UpdateGateBias:linkForward(UpdateGateActivation)
	
	InputCandidateLinear:linkForward(CandidateAdd)
	
	CandidateInputHolder:linkForward(CandidateMultiply)
	
	ResetGateActivation:linkForward(CandidateMultiply)
	
	CandidateMultiply:linkForward(HiddenCandidateLinear)

	HiddenCandidateLinear:linkForward(CandidateAdd)

	CandidateAdd:linkForward(CandidateBias)

	CandidateBias:linkForward(CandidateActivation)
	
	OutputNullaryFunctionHolder:linkForward(OutputSubtract)
	
	UpdateGateActivation:linkForward(OutputSubtract)
	
	OutputSubtract:linkForward(OutputMultiply1)
	
	OutputInputHolder:linkForward(OutputMultiply1)
	
	OutputMultiply1:linkForward(OutputAdd)
	
	UpdateGateActivation:linkForward(OutputMultiply2)

	CandidateActivation:linkForward(OutputMultiply2)
	
	OutputMultiply2:linkForward(OutputAdd)
	
	NewGatedRecurrentUnitCellContainer.inputDimensionSize = inputDimensionSize

	NewGatedRecurrentUnitCellContainer.hiddenDimensionSize = hiddenDimensionSize
	
	NewGatedRecurrentUnitCellContainer.InputResetGateLinear = InputResetGateLinear

	NewGatedRecurrentUnitCellContainer.HiddenResetGateLinear = HiddenResetGateLinear

	NewGatedRecurrentUnitCellContainer.ResetGateBias = ResetGateBias

	NewGatedRecurrentUnitCellContainer.ResetGateAdd = ResetGateAdd

	NewGatedRecurrentUnitCellContainer.ResetGateActivation = ResetGateActivation

	NewGatedRecurrentUnitCellContainer.InputUpdateGateLinear = InputUpdateGateLinear

	NewGatedRecurrentUnitCellContainer.HiddenUpdateGateLinear = HiddenUpdateGateLinear

	NewGatedRecurrentUnitCellContainer.UpdateGateBias = UpdateGateBias

	NewGatedRecurrentUnitCellContainer.UpdateGateAdd = UpdateGateAdd

	NewGatedRecurrentUnitCellContainer.UpdateGateActivation = UpdateGateActivation

	NewGatedRecurrentUnitCellContainer.InputCandidateLinear = InputCandidateLinear

	NewGatedRecurrentUnitCellContainer.HiddenCandidateLinear = HiddenCandidateLinear

	NewGatedRecurrentUnitCellContainer.CandidateBias = CandidateBias
	
	NewGatedRecurrentUnitCellContainer.CandidateMultiply = CandidateMultiply
	
	NewGatedRecurrentUnitCellContainer.CandidateInputHolder = CandidateInputHolder

	NewGatedRecurrentUnitCellContainer.CandidateAdd = CandidateAdd

	NewGatedRecurrentUnitCellContainer.CandidateActivation = CandidateActivation
	
	NewGatedRecurrentUnitCellContainer.OutputNullaryFunctionHolder = OutputNullaryFunctionHolder
	
	NewGatedRecurrentUnitCellContainer.OutputInputHolder = OutputInputHolder
	
	NewGatedRecurrentUnitCellContainer.OutputSubtract = OutputSubtract
	
	NewGatedRecurrentUnitCellContainer.OutputMultiply1 = OutputMultiply1

	NewGatedRecurrentUnitCellContainer.OutputMultiply2 = OutputMultiply2

	NewGatedRecurrentUnitCellContainer.OutputAdd = OutputAdd
	
	NewGatedRecurrentUnitCellContainer.OutputBlockArray = {OutputAdd}
	
	NewGatedRecurrentUnitCellContainer.WeightBlockArray = {InputResetGateLinear, HiddenResetGateLinear, ResetGateBias, InputUpdateGateLinear, HiddenUpdateGateLinear, UpdateGateBias, InputCandidateLinear, HiddenCandidateLinear, CandidateBias}
	
	NewGatedRecurrentUnitCellContainer.ClassesList = parameterDictionary.ClassesList or {}

	NewGatedRecurrentUnitCellContainer.cutOffValue = parameterDictionary.cutOffValue or defaultCutOffValue
	
	NewGatedRecurrentUnitCellContainer:setForwardPropagateFunction(function(featureTensor, hiddenStateTensor)

		local OutputAdd = NewGatedRecurrentUnitCellContainer.OutputAdd

		NewGatedRecurrentUnitCellContainer.ResetGateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		NewGatedRecurrentUnitCellContainer.UpdateGateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		NewGatedRecurrentUnitCellContainer.CandidateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		OutputAdd:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewGatedRecurrentUnitCellContainer.InputResetGateLinear:transform(featureTensor)
		
		NewGatedRecurrentUnitCellContainer.HiddenResetGateLinear:transform(hiddenStateTensor)
		
		NewGatedRecurrentUnitCellContainer.InputUpdateGateLinear:transform(featureTensor)

		NewGatedRecurrentUnitCellContainer.HiddenUpdateGateLinear:transform(hiddenStateTensor)
		
		NewGatedRecurrentUnitCellContainer.InputCandidateLinear:transform(featureTensor)
		
		NewGatedRecurrentUnitCellContainer.CandidateInputHolder:transform(hiddenStateTensor)
		
		NewGatedRecurrentUnitCellContainer.OutputNullaryFunctionHolder:transform()
		
		NewGatedRecurrentUnitCellContainer.OutputInputHolder:transform(hiddenStateTensor)

		local transformedTensor = OutputAdd:waitForTransformedTensor()

		return transformedTensor
		
	end)
	
	NewGatedRecurrentUnitCellContainer:setConvertToClassTensorFunction(function(transformedTensorArray)

		return NewGatedRecurrentUnitCellContainer:convertToClassTensor(transformedTensorArray[1], NewGatedRecurrentUnitCellContainer.ClassesList, NewGatedRecurrentUnitCellContainer.cutOffValue)

	end)
	
	return NewGatedRecurrentUnitCellContainer

end

function GatedRecurrentUnitCellContainer:setCutOffValue(cutOffValue)

	self.cutOffValue = cutOffValue

end

function GatedRecurrentUnitCellContainer:getCutOffValue()

	return self.cutOffValue

end

function GatedRecurrentUnitCellContainer:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function GatedRecurrentUnitCellContainer:getClassesList()

	return self.ClassesList

end

return GatedRecurrentUnitCellContainer
