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

local MinimumBlock = {}

MinimumBlock.__index = MinimumBlock

setmetatable(MinimumBlock, BaseOperatorBlock)

function MinimumBlock.new()

	local NewMinimumBlock = BaseOperatorBlock.new()

	setmetatable(NewMinimumBlock, MinimumBlock)

	NewMinimumBlock:setName("Minimum")
	
	NewMinimumBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewMinimumBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:applyFunction(math.min, table.unpack(inputTensorArray))
	
	end)

	NewMinimumBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		for i, inputTensor in ipairs(inputTensorArray) do
			
			local functionToApply = function(initialPartialFirstDerivativeValue, ...)

				local isMinimum = false

				local lowestValue = -math.huge

				for j, value in ipairs(...) do

					if (value <= lowestValue) then

						isMinimum = (i == j)

						lowestValue = value

					end

				end

				return (isMinimum and initialPartialFirstDerivativeValue) or 0

			end
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			local firstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, initialPartialFirstDerivativeTensor, table.unpack(inputTensorArray))

			chainRuleFirstDerivativeTensorArray[i] = NewMinimumBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewMinimumBlock

end

return MinimumBlocksetmetatable(MinimumBlock, BaseOperatorBlock)

function MinimumBlock.new()

	local NewMinimumBlock = BaseOperatorBlock.new()

	setmetatable(NewMinimumBlock, MinimumBlock)

	NewMinimumBlock:setName("Minimum")
	
	NewMinimumBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewMinimumBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:applyFunction(math.min, table.unpack(inputTensorArray))
	
	end)

	NewMinimumBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		for i, inputTensor in ipairs(inputTensorArray) do
			
			local functionToApply = function(initialPartialFirstDerivativeValue, ...)

				local isMinimum = false

				local lowestValue = -math.huge

				for j, value in ipairs(...) do

					if (value <= lowestValue) then

						isMinimum = (i == j)

						lowestValue = value

					end

				end

				return (isMinimum and initialPartialFirstDerivativeValue) or 0

			end
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			local firstDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, initialPartialFirstDerivativeTensor, table.unpack(inputTensorArray))

			chainRuleFirstDerivativeTensorArray[i] = NewMinimumBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewMinimumBlock

end

return MinimumBlock
