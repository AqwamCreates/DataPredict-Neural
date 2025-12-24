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

local ExpandNumberOfDimensionsBlock = {}

ExpandNumberOfDimensionsBlock.__index = ExpandNumberOfDimensionsBlock

setmetatable(ExpandNumberOfDimensionsBlock, BaseExpansionBlocks)

function ExpandNumberOfDimensionsBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewExpandNumberOfDimensionsBlock = BaseExpansionBlocks.new(parameterDictionary)

	setmetatable(NewExpandNumberOfDimensionsBlock, ExpandNumberOfDimensionsBlock)

	NewExpandNumberOfDimensionsBlock:setName("ExpandNumberOfDimensions")
	
	local dimensionSizeToAddArray = parameterDictionary.dimensionSizeToAddArray
	
	if (not dimensionSizeToAddArray) then error("No dimension size array for expanding number of dimensions.") end

	NewExpandNumberOfDimensionsBlock.dimensionSizeToAddArray = dimensionSizeToAddArray

	NewExpandNumberOfDimensionsBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:expandNumberOfDimensions(inputTensorArray[1], NewExpandNumberOfDimensionsBlock.dimensionSizeToAddArray)

	end)

	NewExpandNumberOfDimensionsBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local numberOfDimensionsToSum = #NewExpandNumberOfDimensionsBlock.dimensionSizeToAddArray
		
		local chainRuleFirstDerivativeTensor = initialPartialFirstDerivativeTensor
		
		for i = 1, numberOfDimensionsToSum, 1 do chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, 1)[1] end -- Remove the first dimension as it is redundant and does not carry any values. If it is not removed, this tensor might broadcast its dimension size elsewhere like during the gradient descent.

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewExpandNumberOfDimensionsBlock

end

function ExpandNumberOfDimensionsBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimensionSizeToAddArray = parameterDictionary.dimensionSizeToAddArray or self.dimensionSizeToAddArray

end

return ExpandNumberOfDimensionsBlock
