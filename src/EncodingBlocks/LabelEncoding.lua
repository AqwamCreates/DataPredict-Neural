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

local BaseEncodingBlock = require(script.Parent.BaseEncodingBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

LabelEncodingBlock = {}

LabelEncodingBlock.__index = LabelEncodingBlock

setmetatable(LabelEncodingBlock, BaseEncodingBlock)

function LabelEncodingBlock.new(parameterDictionary)

	local NewLabelEncodingBlock = BaseEncodingBlock.new()

	setmetatable(NewLabelEncodingBlock, LabelEncodingBlock)

	NewLabelEncodingBlock:setName("LabelEncodingBlock")

	local valueDictionary = parameterDictionary.valueDictionary

	if (not valueDictionary) then error("No value dictionary to generate the tensor.") end

	NewLabelEncodingBlock.valueDictionary = valueDictionary

	NewLabelEncodingBlock:setFunction(function(inputTensorArray)

		local valueDictionary = NewLabelEncodingBlock.valueDictionary

		local functionToApply = function(x) return (valueDictionary[x] or 0) end

		return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	end)

	NewLabelEncodingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		return {initialPartialFirstDerivativeTensor}

	end)

	return NewLabelEncodingBlock

end

function LabelEncodingBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.valueDictionary = parameterDictionary.valueDictionary or self.valueDictionary

end

return LabelEncodingBlock