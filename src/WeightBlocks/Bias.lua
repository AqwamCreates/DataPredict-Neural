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

local BiasBlock = {}

BiasBlock.__index = BiasBlock

setmetatable(BiasBlock, BaseWeightBlock)

function BiasBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewBiasBlock = BaseWeightBlock.new(parameterDictionary)

	setmetatable(NewBiasBlock, BiasBlock)

	NewBiasBlock:setName("Bias")
	
	NewBiasBlock:setFunction(function(inputTensorArray)
		
		local weightTensor = NewBiasBlock:getWeightTensor(true)

		if (not weightTensor) then

			weightTensor = NewBiasBlock:generateWeightTensor()

			NewBiasBlock:setWeightTensor(weightTensor, true)

		end
		
		return AqwamTensorLibrary:add(inputTensorArray[1], weightTensor)
	
	end)

	NewBiasBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		return {initialPartialFirstDerivativeTensor}
		
	end)
	
	NewBiasBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor) -- Since the bias tensor can be broadcasted, we need to sum the dimensions that are the result of the broadcasting so that it matches the original bias tensor size.
		
		local weightTensor = NewBiasBlock:getWeightTensor(true)

		if (not weightTensor) then

			weightTensor = NewBiasBlock:generateWeightTensor()

			NewBiasBlock:setWeightTensor(weightTensor, true)

		end
		
		local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightTensor)

		local firstDerivativeTensor = NewBiasBlock:collapseTensor(initialPartialFirstDerivativeTensor, weightTensorDimensionSizeArray)
		
		return {firstDerivativeTensor}
		
	end)

	return NewBiasBlock

end

return BiasBlock
