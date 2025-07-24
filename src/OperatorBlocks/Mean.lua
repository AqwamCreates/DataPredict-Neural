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

local BaseOperatorBlock = require(script.Parent.BaseOperatorBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

MeanBlock = {}

MeanBlock.__index = MeanBlock

setmetatable(MeanBlock, BaseOperatorBlock)

function MeanBlock.new(parameterDictionary)

	local NewMeanBlock = BaseOperatorBlock.new()

	setmetatable(NewMeanBlock, MeanBlock)

	NewMeanBlock:setName("Mean")

	NewMeanBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	local dimension = parameterDictionary.dimension

	if (not dimension) then error("No dimension is selected.") end

	NewMeanBlock.dimension = dimension

	NewMeanBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:mean(inputTensorArray[1], NewMeanBlock.dimension)

	end)

	NewMeanBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])[NewMeanBlock.dimension]

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(initialPartialFirstDerivativeTensor, dimensionSize)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewMeanBlock

end

function MeanBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimension = parameterDictionary.dimension or self.dimension

end

return MeanBlock