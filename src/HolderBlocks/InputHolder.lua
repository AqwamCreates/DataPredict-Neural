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

local BaseHolderBlock = require(script.Parent.BaseHolderBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

InputHolderBlock = {}

InputHolderBlock.__index = InputHolderBlock

setmetatable(InputHolderBlock, BaseHolderBlock)

function InputHolderBlock.new()

	local NewInputHolderBlock = BaseHolderBlock.new()

	setmetatable(NewInputHolderBlock, InputHolderBlock)

	NewInputHolderBlock:setName("InputHolder")
	
	NewInputHolderBlock:setFunction(function(inputTensorArray)
		
		return inputTensorArray[1]
	
	end)

	NewInputHolderBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		return {initialPartialFirstDerivativeTensor}
		
	end)

	return NewInputHolderBlock

end

return InputHolderBlock