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

local BaseShapeTransformationBlock = require(script.Parent.BaseShapeTransformationBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

PermuteBlock = {}

PermuteBlock.__index = PermuteBlock

setmetatable(PermuteBlock, BaseShapeTransformationBlock)

local function createOriginalDimensionArray(targetDimensionArray)

	local originalDimensionArray = {}

	local originalDimension = 1

	for i, targetDimension in ipairs(targetDimensionArray) do

		originalDimensionArray[targetDimension] = originalDimension

		originalDimension = originalDimension + 1

	end

	return originalDimensionArray

end

function PermuteBlock.new(parameterDictionary)

	local NewPermuteBlock = BaseShapeTransformationBlock.new()

	setmetatable(NewPermuteBlock, PermuteBlock)

	NewPermuteBlock:setName("Permute")

	parameterDictionary = parameterDictionary or {}

	local dimensionArray = parameterDictionary.dimensionArray

	if (not dimensionArray) then error("No dimension array for permuting tensor.") end

	NewPermuteBlock.dimensionArray = dimensionArray

	NewPermuteBlock.originalDimensionArray = createOriginalDimensionArray(dimensionArray)

	NewPermuteBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:permute(inputTensorArray[1], NewPermuteBlock.dimensionArray)

	end)

	NewPermuteBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:permute(initialPartialFirstDerivativeTensor, NewPermuteBlock.originalDimensionArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewPermuteBlock

end

function PermuteBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local dimensionArray = parameterDictionary.dimensionArray

	self.dimensionArray = dimensionArray or self.dimensionArray

	if (dimensionArray) then self.originalDimensionArray = createOriginalDimensionArray(dimensionArray) end

end

return PermuteBlock