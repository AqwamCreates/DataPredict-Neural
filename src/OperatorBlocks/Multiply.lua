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

local MultiplyBlock = {}

MultiplyBlock.__index = MultiplyBlock

setmetatable(MultiplyBlock, BaseOperatorBlock)

function MultiplyBlock.new()

	local NewMultiplyBlock = BaseOperatorBlock.new()

	setmetatable(NewMultiplyBlock, MultiplyBlock)

	NewMultiplyBlock:setName("Multiply")

	NewMultiplyBlock:setFirstDerivativeFunctionRequiresTransformedTensor(true)

	NewMultiplyBlock:setFunction(function(inputTensorArray)
		
		--warn(inputTensorArray[1], inputTensorArray[2])
		
		--warn(NewMultiplyBlock.PreviousFunctionBlockArray[1].name, NewMultiplyBlock.PreviousFunctionBlockArray[2].name)
		
		return AqwamTensorLibrary:multiply(table.unpack(inputTensorArray))

	end)

	NewMultiplyBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local chainRuleFirstDerivativeTensorArray = {}

		for i, inputTensor in ipairs(inputTensorArray) do

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			chainRuleFirstDerivativeTensorArray[i] = NewMultiplyBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray

	end)

	return NewMultiplyBlock

end

return MultiplyBlock
