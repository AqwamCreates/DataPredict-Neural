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

ClampBlock = {}

ClampBlock.__index = ClampBlock

setmetatable(ClampBlock, BaseOperatorBlock)

function ClampBlock.new(parameterDictionary)
	
	local NewClampBlock = BaseOperatorBlock.new()

	setmetatable(NewClampBlock, ClampBlock)

	NewClampBlock:setName("Clamp")
	
	NewClampBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	parameterDictionary = parameterDictionary or {}
	
	NewClampBlock.lowerBoundTensor = parameterDictionary.lowerBoundTensor or -math.huge
	
	NewClampBlock.upperBoundTensor = parameterDictionary.upperBoundTensor or math.huge
	
	NewClampBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:applyFunction(math.clamp, inputTensorArray[1], NewClampBlock.lowerBoundTensor, NewClampBlock.upperBoundTensor)
	
	end)

	NewClampBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local functionToApply = function(value, initialPartialFirstDerivativeValue, lowerBoundValue, upperBoundValue) if ((value >= lowerBoundValue) and (value <= upperBoundValue)) then return value else return 0 end end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1], initialPartialFirstDerivativeTensor, NewClampBlock.lowerBoundTensor, NewClampBlock.upperBoundTensor)
		
		chainRuleFirstDerivativeTensor = NewClampBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		return {chainRuleFirstDerivativeTensor}
		
	end)

	return NewClampBlock

end

return ClampBlock