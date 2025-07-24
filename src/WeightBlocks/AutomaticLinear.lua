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

AutomaticLinearBlock = {}

AutomaticLinearBlock.__index = AutomaticLinearBlock

setmetatable(AutomaticLinearBlock, BaseWeightBlock)

function AutomaticLinearBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAutomaticLinearBlock = BaseWeightBlock.new(parameterDictionary)

	setmetatable(NewAutomaticLinearBlock, AutomaticLinearBlock)

	NewAutomaticLinearBlock:setName("AutomaticLinear")

	local finalDimensionSize = parameterDictionary.finalDimensionSize

	if (not finalDimensionSize) then error("No final dimension size is set.") end

	NewAutomaticLinearBlock.finalDimensionSize = finalDimensionSize

	NewAutomaticLinearBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local weightTensor = NewAutomaticLinearBlock:getWeightTensor(true)

		if (not weightTensor) then

			local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			local inputTensorNumberOfDimensions = #inputTensorDimensionSizeArray

			local dimensionSizeArray = {}

			for i = 2, inputTensorNumberOfDimensions, 1 do table.insert(dimensionSizeArray, inputTensorDimensionSizeArray[i]) end

			table.insert(dimensionSizeArray, NewAutomaticLinearBlock.finalDimensionSize)

			weightTensor = NewAutomaticLinearBlock:generateWeightTensor(dimensionSizeArray)

			NewAutomaticLinearBlock:setWeightTensor(weightTensor, true)

		end

		return AqwamTensorLibrary:dotProduct(inputTensor, weightTensor)

	end)

	NewAutomaticLinearBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local weightTensor = NewAutomaticLinearBlock:getWeightTensor()

		if (not weightTensor) then

			local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			local inputTensorNumberOfDimensions = #inputTensorDimensionSizeArray

			local dimensionSizeArray = {}

			for i = 2, (inputTensorNumberOfDimensions - 1), 1 do table.insert(dimensionSizeArray, inputTensorDimensionSizeArray[i]) end

			table.insert(dimensionSizeArray, parameterDictionary.finalDimensionSize)

			weightTensor = NewAutomaticLinearBlock:generateWeightTensor(dimensionSizeArray)

			NewAutomaticLinearBlock:setWeightTensor(weightTensor, true)

		end

		local weightTensorNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(weightTensor)
		
		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(initialPartialFirstDerivativeTensor, AqwamTensorLibrary:transpose(weightTensor, {weightTensorNumberOfDimensions - 1, weightTensorNumberOfDimensions}))

		return {chainRuleFirstDerivativeTensor}

	end)

	NewAutomaticLinearBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]

		local inputTensorNumberOfDimensions =  AqwamTensorLibrary:getNumberOfDimensions(inputTensor)

		local transposedInputTensor = AqwamTensorLibrary:transpose(inputTensor, {inputTensorNumberOfDimensions - 1, inputTensorNumberOfDimensions})

		local firstDerivativeTensor = AqwamTensorLibrary:dotProduct(transposedInputTensor, initialPartialFirstDerivativeTensor)
		
		local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(NewAutomaticLinearBlock.weightTensor)

		firstDerivativeTensor = NewAutomaticLinearBlock:collapseTensor(firstDerivativeTensor, weightTensorDimensionSizeArray)

		return {firstDerivativeTensor}

	end)

	return NewAutomaticLinearBlock

end

return AutomaticLinearBlock
