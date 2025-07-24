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

local BaseEncodingBlock = require(script.Parent.BaseEncodingBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

PositionalEncodingBlock = {}

PositionalEncodingBlock.__index = PositionalEncodingBlock

setmetatable(PositionalEncodingBlock, BaseEncodingBlock)

local defaultNValue = 10000

local function getPositionalEncodingBlockTensor(sequenceLength, k, n)

	local positionalEncodingTensor = {}

	for i = 1, sequenceLength, 2 do

		local exponent = ((2 * i) / sequenceLength)

		local denominator = math.pow(n, exponent)

		positionalEncodingTensor[i] = math.sin(k / denominator)

		positionalEncodingTensor[i + 1] = math.cos(k / denominator)

	end

	return positionalEncodingTensor

end

local function getPositionalEncodingBlockTensorRecursive(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, finalDimensionSize, nValues)

	local nextDimension = currentDimension + 1
	
	local newTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do newTensor[i] = getPositionalEncodingBlockTensorRecursive(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, finalDimensionSize, nValues) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do newTensor[i] = getPositionalEncodingBlockTensor(finalDimensionSize, i, nValues) end

	end

	return newTensor

end

function PositionalEncodingBlock.new(parameterDictionary)

	local NewPositionalEncodingBlock = BaseEncodingBlock.new()

	setmetatable(NewPositionalEncodingBlock, PositionalEncodingBlock)

	NewPositionalEncodingBlock:setName("PositionalEncodingBlock")

	local nValue = parameterDictionary.nValue or defaultNValue

	local sequenceLength = parameterDictionary.sequenceLength

	if (not sequenceLength) then error("No sequence length to generate the tensor.") end

	if ((sequenceLength % 2) ~= 0) then error("The sequence length must be an even number.") end

	NewPositionalEncodingBlock.nValue = nValue

	NewPositionalEncodingBlock.sequenceLength = sequenceLength

	NewPositionalEncodingBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #dimensionSizeArray

		local transformedTensor = getPositionalEncodingBlockTensorRecursive(inputTensor, dimensionSizeArray, numberOfDimensions, 1, NewPositionalEncodingBlock.sequenceLength, NewPositionalEncodingBlock.nValue)

		return transformedTensor

	end)

	NewPositionalEncodingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		return {initialPartialFirstDerivativeTensor}

	end)

	return NewPositionalEncodingBlock

end

function PositionalEncodingBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.nValue = parameterDictionary.nValue or self.nValue

	local sequenceLength = parameterDictionary.sequenceLength

	if sequenceLength then

		if ((sequenceLength % 2) ~= 0) then error("The final dimension size must be an even number.") end

		self.sequenceLength = sequenceLength

	end

end

return PositionalEncodingBlock