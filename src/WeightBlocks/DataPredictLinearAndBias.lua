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

AutomaticBiasBlock = {}

AutomaticBiasBlock.__index = AutomaticBiasBlock

setmetatable(AutomaticBiasBlock, BaseWeightBlock)

local defaultShareBiasDimensionArray = {1}

function AutomaticBiasBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAutomaticBiasBlock = BaseWeightBlock.new(parameterDictionary)

	setmetatable(NewAutomaticBiasBlock, AutomaticBiasBlock)

	NewAutomaticBiasBlock:setName("AutomaticBias")

	local numberOfDimensions = parameterDictionary.numberOfDimensions

	local shareBiasDimensionArray = parameterDictionary.shareBiasDimensionArray or defaultShareBiasDimensionArray

	if (not numberOfDimensions) then error("The number of dimensions for the bias function block is not set.") end

	if (numberOfDimensions <= 0) then error("The number of dimensions for the bias function block cannot be less than or equal to zero.") end

	if (type(shareBiasDimensionArray) ~= "table") then error("The share bias dimension array is not a table.") end

	for i, dimension in ipairs(shareBiasDimensionArray) do

		if (dimension <= 0) then

			error("The share bias contains a dimension that is less than or equal to zero at the index " .. i .. ".")

		elseif (dimension > numberOfDimensions) then

			error("The share bias contains a dimension that is greater than number of dimensions at the index " .. i .. ".")

		end

	end

	NewAutomaticBiasBlock.numberOfDimensions = numberOfDimensions

	NewAutomaticBiasBlock.shareBiasDimensionArray = shareBiasDimensionArray

	NewAutomaticBiasBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local weightTensor = NewAutomaticBiasBlock:getWeightTensor(true)

		if (not weightTensor) then

			local numberOfDimensions = NewAutomaticBiasBlock.numberOfDimensions

			local shareBiasDimensionArray = NewAutomaticBiasBlock.shareBiasDimensionArray

			local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

			local numberOfDimensionsForInputTensor = #inputTensorDimensionSizeArray

			if (numberOfDimensions > numberOfDimensionsForInputTensor) then error("The number of dimensions for the bias function block exceeds the number of dimensions for the input tensor.") end

			local weightTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

			for _, dimension in ipairs(shareBiasDimensionArray) do weightTensorDimensionSizeArray[dimension] = 1 end

			weightTensor = NewAutomaticBiasBlock:generateWeightTensor(weightTensorDimensionSizeArray)

			NewAutomaticBiasBlock:setWeightTensor(weightTensor, true)

		end

		return AqwamTensorLibrary:add(inputTensor, weightTensor)

	end)

	NewAutomaticBiasBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensor)

		return {initialPartialFirstDerivativeTensor}

	end)

	NewAutomaticBiasBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor) -- Since the bias tensor can be broadcasted, we need to sum the dimensions that are the result of the broadcasting so that it matches the original bias tensor size.

		local weightTensor = NewAutomaticBiasBlock:getWeightTensor(true)

		local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightTensor)

		local firstDerivativeTensor = NewAutomaticBiasBlock:collapseTensor(initialPartialFirstDerivativeTensor, weightTensorDimensionSizeArray)

		return {firstDerivativeTensor}

	end)

	return NewAutomaticBiasBlock

end

return AutomaticBiasBlock
