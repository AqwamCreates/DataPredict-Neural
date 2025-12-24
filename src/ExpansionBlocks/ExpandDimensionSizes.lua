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

local BaseExpansionBlocks = require(script.Parent.BaseExpansionBlocks)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ExpandDimensionSizesBlock = {}

ExpandDimensionSizesBlock.__index = ExpandDimensionSizesBlock

setmetatable(ExpandDimensionSizesBlock, BaseExpansionBlocks)

function ExpandDimensionSizesBlock.new(parameterDictionary)

	local NewExpandDimensionSizesBlock = BaseExpansionBlocks.new()

	setmetatable(NewExpandDimensionSizesBlock, ExpandDimensionSizesBlock)

	NewExpandDimensionSizesBlock:setName("ExpandDimensionSizesBlock")

	parameterDictionary = parameterDictionary or {}

	local targetDimensionSizeArray = parameterDictionary.targetDimensionSizeArray

	if (not targetDimensionSizeArray) then error("No target dimension size array for expanding dimension sizes.") end

	NewExpandDimensionSizesBlock.targetDimensionSizeArray = targetDimensionSizeArray

	NewExpandDimensionSizesBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:expandDimensionSizes(inputTensorArray[1], NewExpandDimensionSizesBlock.targetDimensionSizeArray)

	end)

	NewExpandDimensionSizesBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local targetDimensionSizeArray = NewExpandDimensionSizesBlock.targetDimensionSizeArray
		
		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local chainRuleFirstDerivativeTensor = initialPartialFirstDerivativeTensor
		
		for dimension, dimensionSize in ipairs(inputTensorDimensionSizeArray) do
			
			if (dimensionSize == 1) and (targetDimensionSizeArray[dimension] > 1) then
				
				chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, dimension)
				
			end
			
		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewExpandDimensionSizesBlock

end

function ExpandDimensionSizesBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.targetDimensionSizeArray = parameterDictionary.targetDimensionSizeArray or self.targetDimensionSizeArray

end

return ExpandDimensionSizesBlock
