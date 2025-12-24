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

local BaseShapeTransformationBlock = require(script.Parent.BaseShapeTransformationBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local TransposeBlock = {}

TransposeBlock.__index = TransposeBlock

setmetatable(TransposeBlock, BaseShapeTransformationBlock)

function TransposeBlock.new(parameterDictionary)

	local NewTransposeBlock = BaseShapeTransformationBlock.new()

	setmetatable(NewTransposeBlock, TransposeBlock)

	NewTransposeBlock:setName("Transpose")

	parameterDictionary = parameterDictionary or {}

	local dimensionArray = parameterDictionary.dimensionArray

	if (not dimensionArray) then error("No dimension array for transposing tensor.") end

	NewTransposeBlock.dimensionArray = dimensionArray

	NewTransposeBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:transpose(inputTensorArray[1], NewTransposeBlock.dimensionArray)

	end)

	NewTransposeBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:transpose(initialPartialFirstDerivativeTensor, NewTransposeBlock.dimensionArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewTransposeBlock

end

function TransposeBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimensionArray = parameterDictionary.dimensionArray or self.dimensionArray

end

return TransposeBlock
