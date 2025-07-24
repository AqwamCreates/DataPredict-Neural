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

local BaseHolderBlock = require(script.Parent.BaseHolderBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

VariableHolderBlock = {}

VariableHolderBlock.__index = VariableHolderBlock

setmetatable(VariableHolderBlock, BaseHolderBlock)

local defaultLearningRate = 0.01

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

end

function VariableHolderBlock.new(parameterDictionary)

	local NewVariableHolderBlock = BaseHolderBlock.new()

	setmetatable(NewVariableHolderBlock, VariableHolderBlock)

	NewVariableHolderBlock:setName("VariableHolder")
	
	NewVariableHolderBlock:setRequiresInputTensors(false)
	
	NewVariableHolderBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	local variableTensor = parameterDictionary.variableTensor
	
	if (not variableTensor) then error("No variable tensor.") end
	
	NewVariableHolderBlock.variableTensor = variableTensor
	
	NewVariableHolderBlock.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewVariableHolderBlock.Optimizer = parameterDictionary.Optimizer

	NewVariableHolderBlock.Regularizer = parameterDictionary.Regularizer
	
	NewVariableHolderBlock:setFunction(function()
		
		return NewVariableHolderBlock.variableTensor
	
	end)
	
	NewVariableHolderBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(NewVariableHolderBlock.variableTensor, initialPartialFirstDerivativeTensor)

		return {chainRuleFirstDerivativeTensor}
		
	end)

	NewVariableHolderBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor)

		local numberOfDimensionsOfInitialPartialFirstDerivativeTensor = #AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(NewVariableHolderBlock.variableTensor)

		local numberOfDimensionsOfWeightTensor = #weightTensorDimensionSizeArray

		local numberOfDimensionsToSum = numberOfDimensionsOfInitialPartialFirstDerivativeTensor - numberOfDimensionsOfWeightTensor

		local firstDerivativeTensor = initialPartialFirstDerivativeTensor

		for i = 1, numberOfDimensionsToSum, 1 do firstDerivativeTensor = AqwamTensorLibrary:sum(firstDerivativeTensor, 1)[1] end -- Remove the first dimension as it is redundant and does not carry any values. If it is not removed, this tensor might broadcast its dimension size elsewhere like during the gradient descent.

		for i, size in ipairs(weightTensorDimensionSizeArray) do

			if (size == 1) then firstDerivativeTensor = AqwamTensorLibrary:sum(firstDerivativeTensor, i) end

		end

		return {firstDerivativeTensor}
		
	end)

	return NewVariableHolderBlock

end

function VariableHolderBlock:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.variableTensor = parameterDictionary.variableTensor or self.variableTensor

	self.learningRate = parameterDictionary.learningRate or self.learningRate

	self.Optimizer = parameterDictionary.Optimizer or self.Optimizer

	self.Regularizer = parameterDictionary.Regularizer or self.Regularizer
	
end

function VariableHolderBlock:gradientDescent(lossTensor, numberOfData)

	local variableTensor = self.variableTensor

	local learningRate = self.learningRate

	local Optimizer = self.Optimizer

	local Regularizer = self.Regularizer

	if (Regularizer) then

		local regularizationTensor = Regularizer:calculate(variableTensor)

		lossTensor = AqwamTensorLibrary:add(lossTensor, regularizationTensor)

	end

	if (numberOfData ~= nil) and (numberOfData ~= 1) then 

		lossTensor = AqwamTensorLibrary:divide(lossTensor, numberOfData) 

	end

	if (Optimizer) then

		lossTensor = Optimizer:calculate(learningRate, lossTensor)

	else

		lossTensor = AqwamTensorLibrary:multiply(learningRate, lossTensor)

	end

	self.variableTensor = AqwamTensorLibrary:subtract(variableTensor, lossTensor)

end

function VariableHolderBlock:setVariableTensor(variableTensor, doNotDeepCopy)

	if (doNotDeepCopy) then
		
		self.variableTensor = variableTensor
		
	else
		
		self.variableTensor = deepCopyTable(variableTensor)
		
	end

end

function VariableHolderBlock:getVariableTensor(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.variableTensor

	else

		return deepCopyTable(self.variableTensor)

	end

end

function VariableHolderBlock:setLearningRate(learningRate)

	self.learningRate = learningRate or defaultLearningRate

end

function VariableHolderBlock:getLearningRate()

	return self.learningRate

end

function VariableHolderBlock:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function VariableHolderBlock:getOptimizer()

	return self.Optimizer

end

function VariableHolderBlock:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function VariableHolderBlock:getRegularizer()

	return self.Regularizer

end

return VariableHolderBlock