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

local ReshapeBlock = {}

ReshapeBlock.__index = ReshapeBlock

setmetatable(ReshapeBlock, BaseShapeTransformationBlock)

function ReshapeBlock.new(parameterDictionary)

	local NewReshapeBlock = BaseShapeTransformationBlock.new()

	setmetatable(NewReshapeBlock, ReshapeBlock)

	NewReshapeBlock:setName("Reshape")

	parameterDictionary = parameterDictionary or {}

	local dimensionSizeArray = parameterDictionary.dimensionSizeArray

	if (not dimensionSizeArray) then error("No dimension size array to reshape the tensor.") end

	NewReshapeBlock.dimensionSizeArray = dimensionSizeArray

	NewReshapeBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:reshape(inputTensorArray[1], NewReshapeBlock.dimensionSizeArray)

	end)

	NewReshapeBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:reshape(initialPartialFirstDerivativeTensor, inputTensorDimensionSizeArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewReshapeBlock

end

function ReshapeBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimensionSizeArray = parameterDictionary.dimensionSizeArray or self.dimensionSizeArray

end

return ReshapeBlock
