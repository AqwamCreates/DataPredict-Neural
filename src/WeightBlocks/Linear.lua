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

local BaseWeightBlock = require(script.Parent.BaseWeightBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local LinearBlock = {}

LinearBlock.__index = LinearBlock

setmetatable(LinearBlock, BaseWeightBlock)

function LinearBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewLinearBlock = BaseWeightBlock.new(parameterDictionary)

	setmetatable(NewLinearBlock, LinearBlock)

	NewLinearBlock:setName("Linear")

	NewLinearBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewLinearBlock:setFunction(function(inputTensorArray)

		local weightTensor = NewLinearBlock:getWeightTensor(true)

		if (not weightTensor) then

			weightTensor = NewLinearBlock:generateWeightTensor()

			NewLinearBlock:setWeightTensor(weightTensor, true)

		end

		return AqwamTensorLibrary:dotProduct(inputTensorArray[1], weightTensor)

	end)

	NewLinearBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local weightTensor = NewLinearBlock:getWeightTensor(true)

		if (not weightTensor) then

			weightTensor = NewLinearBlock:generateWeightTensor()

			NewLinearBlock:setWeightTensor(weightTensor, true)

		end

		local weightTensorNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(weightTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(initialPartialFirstDerivativeTensor, AqwamTensorLibrary:transpose(weightTensor, {weightTensorNumberOfDimensions - 1, weightTensorNumberOfDimensions}))

		return {chainRuleFirstDerivativeTensor}

	end)

	NewLinearBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]
		
		local inputTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(inputTensor)
		
		local transposedInputTensor = AqwamTensorLibrary:transpose(inputTensor, {inputTensorNumberOfDimensions - 1, inputTensorNumberOfDimensions})
		
		local firstDerivativeTensor = AqwamTensorLibrary:dotProduct(transposedInputTensor, initialPartialFirstDerivativeTensor)
		
		local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(NewLinearBlock.weightTensor)
		
		firstDerivativeTensor = NewLinearBlock:collapseTensor(firstDerivativeTensor, weightTensorDimensionSizeArray)

		return {firstDerivativeTensor}

	end)

	return NewLinearBlock

end

return LinearBlock
