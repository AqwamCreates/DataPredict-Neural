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

local MaximumBlock = {}

MaximumBlock.__index = MaximumBlock

setmetatable(MaximumBlock, BaseOperatorBlock)

function MaximumBlock.new()

	local NewMaximumBlock = BaseOperatorBlock.new()

	setmetatable(NewMaximumBlock, MaximumBlock)

	NewMaximumBlock:setName("Maximum")
	
	NewMaximumBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewMaximumBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:applyFunction(math.max, table.unpack(inputTensorArray))
	
	end)

	NewMaximumBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		for i, inputTensor in ipairs(inputTensorArray) do
			
			local functionToApply = function(initialPartialFirstDerivativeValue, ...)

				local isMaximum = false

				local highestValue = math.huge

				for j, value in ipairs(...) do

					if (value >= highestValue) then

						isMaximum = (i == j)

						highestValue = value

					end

				end

				return (isMaximum and initialPartialFirstDerivativeValue) or 0

			end

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			local firstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, initialPartialFirstDerivativeTensor, table.unpack(inputTensorArray))

			chainRuleFirstDerivativeTensorArray[i] = NewMaximumBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewMaximumBlock

end

return MaximumBlock
