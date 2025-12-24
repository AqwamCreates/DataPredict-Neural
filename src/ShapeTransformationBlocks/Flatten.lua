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

local FlattenBlock = {}

FlattenBlock.__index = FlattenBlock

setmetatable(FlattenBlock, BaseShapeTransformationBlock)

local defaultDimensionArray = {2, math.huge}

function FlattenBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewFlattenBlock = BaseShapeTransformationBlock.new(parameterDictionary)

	setmetatable(NewFlattenBlock, FlattenBlock)

	NewFlattenBlock:setName("Flatten")

	NewFlattenBlock.dimensionArray = parameterDictionary.dimensionArray or defaultDimensionArray

	NewFlattenBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:flatten(inputTensorArray[1], NewFlattenBlock.dimensionArray)

	end)

	NewFlattenBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:reshape(initialPartialFirstDerivativeTensor, dimensionSizeArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewFlattenBlock

end

function FlattenBlock:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.dimensionArray = parameterDictionary.dimensionArray or self.dimensionArray
	
end

return FlattenBlock
