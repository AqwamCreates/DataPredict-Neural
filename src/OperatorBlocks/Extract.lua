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

ExtractBlock = {}

ExtractBlock.__index = ExtractBlock

setmetatable(ExtractBlock, BaseOperatorBlock)

function ExtractBlock.new(parameterDictionary)

	local NewExtractBlock = BaseOperatorBlock.new()

	setmetatable(NewExtractBlock, ExtractBlock)

	NewExtractBlock:setName("Extract")

	NewExtractBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	parameterDictionary = parameterDictionary or {}

	local originDimensionIndexArray = parameterDictionary.originDimensionIndexArray

	local targetDimensionIndexArray = parameterDictionary.targetDimensionIndexArray

	if (not originDimensionIndexArray) then error("No origin dimension index array to extract the tensor.") end

	if (not targetDimensionIndexArray) then error("No target dimension index array to extract the tensor.") end

	NewExtractBlock.originDimensionIndexArray = originDimensionIndexArray

	NewExtractBlock.targetDimensionIndexArray = targetDimensionIndexArray

	NewExtractBlock:setFunction(function(inputTensorArray)

		return AqwamTensorLibrary:extract(inputTensorArray[1], NewExtractBlock.originDimensionIndexArray, NewExtractBlock.targetDimensionIndexArray)

	end)

	NewExtractBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local originDimensionIndexArray = NewExtractBlock.originDimensionIndexArray

		local targetDimensionIndexArray = NewExtractBlock.targetDimensionIndexArray

		local chainRuleFirstDerivativeTensor = initialPartialFirstDerivativeTensor

		local numberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray = {}

		local tailPaddingDimensionSizeArray = {}

		for dimension = 1, numberOfDimensions, 1 do

			headPaddingDimensionSizeArray[dimension] = originDimensionIndexArray[dimension] - 1

			tailPaddingDimensionSizeArray[dimension] = inputTensorDimensionSizeArray[dimension] - targetDimensionIndexArray[dimension]

		end

		for dimension = numberOfDimensions, 1, -1 do

			local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(chainRuleFirstDerivativeTensor)

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				local tensorHeadPaddingDimensionSizeArray = table.clone(firstDerivativeTensorDimensionSizeArray)

				tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

				local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

				chainRuleFirstDerivativeTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, chainRuleFirstDerivativeTensor, dimension)

			end

			if (tailPaddingDimensionSize >= 1) then

				local tensorTailPaddingDimensionSizeArray = table.clone(firstDerivativeTensorDimensionSizeArray)

				tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

				local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

				chainRuleFirstDerivativeTensor = AqwamTensorLibrary:concatenate(chainRuleFirstDerivativeTensor, tailPaddingTensor, dimension)

			end

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewExtractBlock

end

function ExtractBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.originDimensionSizeArray = parameterDictionary.originDimensionSizeArray or self.originDimensionSizeArray

	self.targetDimensionSizeArray = parameterDictionary.targetDimensionSizeArray or self.targetDimensionSizeArray

end

return ExtractBlock