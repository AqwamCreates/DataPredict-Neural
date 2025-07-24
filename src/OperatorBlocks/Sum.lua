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

SumBlock = {}

SumBlock.__index = SumBlock

setmetatable(SumBlock, BaseOperatorBlock)

function SumBlock.new(parameterDictionary)

	local NewSumBlock = BaseOperatorBlock.new()

	setmetatable(NewSumBlock, SumBlock)

	NewSumBlock:setName("Sum")

	NewSumBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewSumBlock.dimension = parameterDictionary.dimension

	NewSumBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:sum(inputTensorArray[1], NewSumBlock.dimension)

	end)

	NewSumBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local chainRuleFirstDerivativeTensor
		
		if (NewSumBlock.dimension) then

			chainRuleFirstDerivativeTensor = AqwamTensorLibrary:expandDimensionSizes(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		else

			chainRuleFirstDerivativeTensor = AqwamTensorLibrary:expandNumberOfDimensions(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewSumBlock

end

function SumBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimension = parameterDictionary.dimension or self.dimension

end

return SumBlock