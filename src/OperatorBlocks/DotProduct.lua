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

local DotProductBlock = {}

DotProductBlock.__index = DotProductBlock

setmetatable(DotProductBlock, BaseOperatorBlock)

function DotProductBlock.new()

	local NewDotProductBlock = BaseOperatorBlock.new()

	setmetatable(NewDotProductBlock, DotProductBlock)

	NewDotProductBlock:setName("DotProduct")

	NewDotProductBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewDotProductBlock:setFunction(function(inputTensorArray)
		
		if (#inputTensorArray ~= 2) then error("Dot product can only take in two inputs.") end

		return AqwamTensorLibrary:dotProduct(inputTensorArray[1], inputTensorArray[2])

	end)

	NewDotProductBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local inputTensor1 = inputTensorArray[1]
		
		local inputTensor2 = inputTensorArray[2]
		
		local inputTensor1NumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(inputTensor1)
		
		local inputTensor2NumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(inputTensor2)
		
		local inputTensor1FirstDerivativeTensor = AqwamTensorLibrary:dotProduct(initialPartialFirstDerivativeTensor, AqwamTensorLibrary:transpose(inputTensor2, {inputTensor2NumberOfDimensions - 1, inputTensor2NumberOfDimensions}))

		local inputTensor2FirstDerivativeTensor = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(inputTensor1, {inputTensor1NumberOfDimensions - 1, inputTensor1NumberOfDimensions}), initialPartialFirstDerivativeTensor)
		
		local inputTensor1DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor1)
		
		local inputTensor2DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor2)
		
		inputTensor1FirstDerivativeTensor = NewDotProductBlock:collapseTensor(inputTensor1FirstDerivativeTensor, inputTensor1DimensionSizeArray)
		
		inputTensor2DimensionSizeArray = NewDotProductBlock:collapseTensor(inputTensor2DimensionSizeArray, inputTensor2DimensionSizeArray)

		return {inputTensor1FirstDerivativeTensor, inputTensor2FirstDerivativeTensor}

	end)

	return NewDotProductBlock

end

return DotProductBlock
