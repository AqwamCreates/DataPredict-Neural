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

SubtractBlock = {}

SubtractBlock.__index = SubtractBlock

setmetatable(SubtractBlock, BaseOperatorBlock)

function SubtractBlock.new()

	local NewSubtractBlock = BaseOperatorBlock.new()

	setmetatable(NewSubtractBlock, SubtractBlock)

	NewSubtractBlock:setName("Subtract")

	NewSubtractBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewSubtractBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:subtract(table.unpack(inputTensorArray))

	end)

	NewSubtractBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local chainRuleFirstDerivativeTensorArray = {}

		for i, inputTensor in ipairs(inputTensorArray) do

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			chainRuleFirstDerivativeTensorArray[i] = NewSubtractBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray

	end)

	return NewSubtractBlock

end

return SubtractBlock