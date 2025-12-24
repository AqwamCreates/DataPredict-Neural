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

local StandardDeviationBlock = {}

StandardDeviationBlock.__index = StandardDeviationBlock

setmetatable(StandardDeviationBlock, BaseOperatorBlock)

function StandardDeviationBlock.new(parameterDictionary)

	local NewStandardDeviationBlock = BaseOperatorBlock.new()

	setmetatable(NewStandardDeviationBlock, StandardDeviationBlock)

	NewStandardDeviationBlock:setName("StandardDeviation")

	NewStandardDeviationBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	local dimension = parameterDictionary.dimension

	if (not dimension) then error("No dimension is selected.") end

	NewStandardDeviationBlock.dimension = dimension

	NewStandardDeviationBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:standardDeviation(inputTensorArray[1], NewStandardDeviationBlock.dimension)

	end)

	NewStandardDeviationBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]
		
		local dimension = NewStandardDeviationBlock.dimension

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)[dimension]

		local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(inputTensor, dimension)

		local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(2, standardDeviationTensor, dimensionSize)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(initialPartialFirstDerivativeTensor, chainRuleFirstDerivativeTensorPart1)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewStandardDeviationBlock

end

function StandardDeviationBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimension = parameterDictionary.dimension or self.dimension

end

return StandardDeviationBlock
