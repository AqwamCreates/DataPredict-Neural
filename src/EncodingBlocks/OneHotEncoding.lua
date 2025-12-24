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

local OneHotEncodingBlock = {}

OneHotEncodingBlock.__index = OneHotEncodingBlock

setmetatable(OneHotEncodingBlock, BaseEncodingBlock)

local defaultOneHotEncodingBlockMode = "Index"

local function createOneHotTensorFromIndex(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, finalDimensionSize)

	local nextDimension = currentDimension + 1

	local oneHotTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do oneHotTensor[i] = createOneHotTensorFromIndex(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, finalDimensionSize) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local subTensor = table.create(finalDimensionSize, 0)

			local index = tensor[i]

			if (type(index) ~= "number") then error("The tensor must only have numbers for one hot encoding conversion from indices.") end

			if (subTensor[index]) then subTensor[index] = 1 end

			oneHotTensor[i] = subTensor

		end

	end

	return oneHotTensor

end

local function createOneHotTensorFromKey(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, finalDimensionSize, indexDictionary)

	local nextDimension = currentDimension + 1

	local oneHotTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do oneHotTensor[i] = createOneHotTensorFromKey(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, finalDimensionSize, indexDictionary) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local subTensor = table.create(finalDimensionSize, 0)

			local key = tensor[i]

			local index = indexDictionary[key]

			if (index) then subTensor[index] = 1 end

			oneHotTensor[i] = subTensor

		end

	end

	return oneHotTensor

end

function OneHotEncodingBlock.new(parameterDictionary)

	local NewOneHotEncodingBlock = BaseEncodingBlock.new()

	setmetatable(NewOneHotEncodingBlock, OneHotEncodingBlock)

	NewOneHotEncodingBlock:setName("OneHotEncodingBlock")

	local finalDimensionSize = parameterDictionary.finalDimensionSize

	local oneHotEncodingMode = parameterDictionary.oneHotEncodingMode or defaultOneHotEncodingBlockMode

	local indexDictionary = parameterDictionary.indexDictionary

	if (not finalDimensionSize) then error("No final dimension size to generate the tensor.") end

	NewOneHotEncodingBlock.finalDimensionSize = finalDimensionSize

	NewOneHotEncodingBlock.oneHotEncodingMode = oneHotEncodingMode

	NewOneHotEncodingBlock.indexDictionary = indexDictionary

	NewOneHotEncodingBlock:setFunction(function(inputTensorArray)

		local finalDimensionSize = NewOneHotEncodingBlock.finalDimensionSize

		local oneHotEncodingMode = NewOneHotEncodingBlock.oneHotEncodingMode

		local inputTensor = inputTensorArray[1]

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #dimensionSizeArray

		local transformedTensor

		if (oneHotEncodingMode == "Index") then

			transformedTensor = createOneHotTensorFromIndex(inputTensor, dimensionSizeArray, numberOfDimensions, 1, finalDimensionSize)

		elseif (oneHotEncodingMode == "Key") then

			local indexDictionary = NewOneHotEncodingBlock.indexDictionary

			if (not indexDictionary) then error("No index dictionary for one hot encoding key mode.") end

			transformedTensor = createOneHotTensorFromKey(inputTensor, dimensionSizeArray, numberOfDimensions, 1, finalDimensionSize, indexDictionary)

		else

			error("Invalid one hot encoding mode.")

		end

		return transformedTensor

	end)

	NewOneHotEncodingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		return {initialPartialFirstDerivativeTensor}

	end)

	return NewOneHotEncodingBlock

end

function OneHotEncodingBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.finalDimensionSize = parameterDictionary.finalDimensionSize or self.finalDimensionSize

	self.oneHotEncodingMode = parameterDictionary.oneHotEncodingMode or self.oneHotEncodingMode

	self.indexDictionary = parameterDictionary.indexDictionary or self.indexDictionary

end

return OneHotEncodingBlock
