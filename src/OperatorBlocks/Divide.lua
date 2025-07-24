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

DivideBlock = {}

DivideBlock.__index = DivideBlock

setmetatable(DivideBlock, BaseOperatorBlock)

function DivideBlock.new()

	local NewDivideBlock = BaseOperatorBlock.new()

	setmetatable(NewDivideBlock, DivideBlock)

	NewDivideBlock:setName("Divide")

	NewDivideBlock:setFirstDerivativeFunctionRequiresTransformedTensor(true)

	NewDivideBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:divide(table.unpack(inputTensorArray))

	end)

	NewDivideBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local chainRuleFirstDerivativeTensorArray = {}

		for i, inputTensor in ipairs(inputTensorArray) do

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			chainRuleFirstDerivativeTensorArray[i] = NewDivideBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray

	end)

	return NewDivideBlock

end

return DivideBlock