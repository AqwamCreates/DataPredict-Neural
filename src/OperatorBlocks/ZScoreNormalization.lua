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

ZScoreNormalizationBlock = {}

ZScoreNormalizationBlock.__index = ZScoreNormalizationBlock

setmetatable(ZScoreNormalizationBlock, BaseOperatorBlock)

function ZScoreNormalizationBlock.new(parameterDictionary)

	local NewZScoreNormalizationBlock = BaseOperatorBlock.new()

	setmetatable(NewZScoreNormalizationBlock, ZScoreNormalizationBlock)

	NewZScoreNormalizationBlock:setName("ZScoreNormalization")

	NewZScoreNormalizationBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	local dimension = parameterDictionary.dimension

	if (not dimension) then error("No dimension is selected.") end

	NewZScoreNormalizationBlock.dimension = dimension

	NewZScoreNormalizationBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:zScoreNormalization(inputTensorArray[1], NewZScoreNormalizationBlock.dimension)

	end)

	NewZScoreNormalizationBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(inputTensorArray[1], NewZScoreNormalizationBlock.dimension)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(initialPartialFirstDerivativeTensor, standardDeviationTensor)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewZScoreNormalizationBlock

end

function ZScoreNormalizationBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimension = parameterDictionary.dimension or self.dimension

end

return ZScoreNormalizationBlock
