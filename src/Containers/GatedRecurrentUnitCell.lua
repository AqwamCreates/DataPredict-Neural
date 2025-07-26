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
	
	if (not InputResetGateLinear) then InputResetGateLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not HiddenResetGateLinear) then HiddenResetGateLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not ResetGateAdd) then ResetGateAdd = require(DataPredictNeural.OperatorBlocks.Add).new() end
	
	if (not ResetGateBias) then ResetGateBias = require(DataPredictNeural.WeightBlocks.Bias).new({dimensionSizeArray = {1, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not ResetGateActivation) then ResetGateActivation = require(DataPredictNeural.ActivationBlocks.Sigmoid).new() end
	
	if (not InputUpdateGateLinear) then InputUpdateGateLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end

	if (not HiddenUpdateGateLinear) then HiddenUpdateGateLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not UpdateGateAdd) then UpdateGateAdd = require(DataPredictNeural.OperatorBlocks.Add).new() end
	
	if (not UpdateGateBias) then UpdateGateBias = require(DataPredictNeural.WeightBlocks.Bias).new({dimensionSizeArray = {1, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not UpdateGateActivation) then UpdateGateActivation = require(DataPredictNeural.ActivationBlocks.Sigmoid).new() end
	
	if (not InputCandidateLinear) then InputCandidateLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {inputDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end

	if (not HiddenCandidateLinear) then HiddenCandidateLinear = require(DataPredictNeural.WeightBlocks.Linear).new({dimensionSizeArray = {hiddenDimensionSize, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not CandidateAdd) then CandidateAdd = require(DataPredictNeural.OperatorBlocks.Add).new() end

	if (not CandidateBias) then CandidateBias = require(DataPredictNeural.WeightBlocks.Bias).new({dimensionSizeArray = {1, hiddenDimensionSize}, learningRate = learningRate, weightInitializationMode = weightInitializationMode}) end
	
	if (not CandidateInputHolder) then CandidateInputHolder = require(DataPredictNeural.HolderBlocks.InputHolder).new() end
	
	if (not CandidateMultiply) then CandidateMultiply = require(DataPredictNeural.OperatorBlocks.Multiply).new() end
	
	if (not CandidateActivation) then CandidateActivation = require(DataPredictNeural.ActivationBlocks.Tanh).new() end
	
	if (not OutputNullaryFunctionHolder) then OutputNullaryFunctionHolder = require(DataPredictNeural.HolderBlocks.NullaryFunctionHolder).new({Function = (function() return 1 end), ChainRuleFirstDerivativeFunction = (function() return end)}) end
	
	if (not OutputInputHolder) then OutputInputHolder = require(DataPredictNeural.HolderBlocks.InputHolder).new() end
	
	if (not OutputSubtract) then OutputSubtract = require(DataPredictNeural.OperatorBlocks.Subtract).new() end
	
	if (not OutputMultiply1) then OutputMultiply1 = require(DataPredictNeural.OperatorBlocks.Multiply).new() end
	
	if (not OutputMultiply2) then OutputMultiply2 = require(DataPredictNeural.OperatorBlocks.Multiply).new() end
	
	if (not OutputAdd) then OutputAdd = require(DataPredictNeural.OperatorBlocks.Add).new() end
	
	--OutputInputHolder.name = "OutputInputHolder"
	
	--OutputNullaryFunctionHolder.name = "OutputNullaryFunctionHolder"
	
	--CandidateInputHolder.name = "CandidateInputHolder"
	
	--CandidateActivation.name = "CandidateActivation"
	
	--ResetGateActivation.name = "ResetGateActivation"
	
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
		
		local ResetGateActivation = NewGatedRecurrentUnitCellContainer.ResetGateActivation

		local UpdateGateActivation = NewGatedRecurrentUnitCellContainer.UpdateGateActivation

		local CandidateActivation = NewGatedRecurrentUnitCellContainer.CandidateActivation

		local OutputAdd = NewGatedRecurrentUnitCellContainer.OutputAdd

		ResetGateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		UpdateGateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		CandidateActivation:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.

		OutputAdd:setTransformedTensor(nil, true) -- To ensure that we don't output old tensor values.
		
		NewGatedRecurrentUnitCellContainer.InputResetGateLinear:transform(featureTensor)
		
		NewGatedRecurrentUnitCellContainer.HiddenResetGateLinear:transform(hiddenStateTensor)
		
		NewGatedRecurrentUnitCellContainer.InputUpdateGateLinear:transform(featureTensor)

		NewGatedRecurrentUnitCellContainer.HiddenUpdateGateLinear:transform(hiddenStateTensor)
		
		NewGatedRecurrentUnitCellContainer.CandidateInputHolder:transform(hiddenStateTensor)
		
		NewGatedRecurrentUnitCellContainer.InputCandidateLinear:transform(featureTensor)
		
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

	self.cutOffValue = cutOffValue or self.cutOffValue

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
