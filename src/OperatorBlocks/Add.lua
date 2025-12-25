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

local AddBlock = {}

AddBlock.__index = AddBlock

setmetatable(AddBlock, BaseOperatorBlock)

function AddBlock.new()

	local NewAddBlock = BaseOperatorBlock.new()

	setmetatable(NewAddBlock, AddBlock)

	NewAddBlock:setName("Add")
	
	NewAddBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewAddBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:add(table.unpack(inputTensorArray))
	
	end)

	NewAddBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		for i, inputTensor in ipairs(inputTensorArray) do
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			chainRuleFirstDerivativeTensorArray[i] = NewAddBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		end

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewAddBlock

end

return AddBlock
