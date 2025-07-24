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

local BasePaddingBlock = require(script.Parent.BasePaddingBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

ConstantPaddingBlock = {}

ConstantPaddingBlock.__index = ConstantPaddingBlock

setmetatable(ConstantPaddingBlock, BasePaddingBlock)

local defaultHeadPaddingDimensionSizeArray = {1, 1}

local defaultTailPaddingDimensionSizeArray = {1, 1}

local defaultValue = 0

local function padArraysToEqualLengths(numberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	local headPaddingNumberOfDimensionsOffset = numberOfDimensions - #headPaddingDimensionSizeArray

	local tailPaddingNumberOfDimensionsOffset = numberOfDimensions - #tailPaddingDimensionSizeArray 

	if (headPaddingNumberOfDimensionsOffset ~= 0) then for i = 1, headPaddingNumberOfDimensionsOffset, 1 do table.insert(headPaddingDimensionSizeArray, 1, 0) end end

	if (tailPaddingNumberOfDimensionsOffset ~= 0) then for i = 1, tailPaddingNumberOfDimensionsOffset, 1 do table.insert(tailPaddingDimensionSizeArray, 1, 0) end end

	return headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray

end

local function getTotalDimensionSize(dimensionSizeArray)
	
	local totalDimensionSize = 1
	
	for _, size in ipairs(dimensionSizeArray) do totalDimensionSize = totalDimensionSize * size end
	
	return totalDimensionSize
	
end

function ConstantPaddingBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewConstantPaddingBlock = BasePaddingBlock.new()

	setmetatable(NewConstantPaddingBlock, ConstantPaddingBlock)

	NewConstantPaddingBlock:setName("ConstantPadding")

	NewConstantPaddingBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewConstantPaddingBlock.headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or defaultHeadPaddingDimensionSizeArray

	NewConstantPaddingBlock.tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or defaultTailPaddingDimensionSizeArray
	
	NewConstantPaddingBlock.value = parameterDictionary.value or defaultValue
	
	local chainRuleFirstDerivativeMultiplierFunction = function(chainRuleFirstDerivativeValue, inputValue, value, chainRuleFirstDerivativeMultiplierValue)
		
		if (inputValue == value) then chainRuleFirstDerivativeValue = chainRuleFirstDerivativeValue * chainRuleFirstDerivativeMultiplierValue end
		
		return chainRuleFirstDerivativeValue
		
	end
	
	local chainRuleFirstDerivativeMultiplierValue

	NewConstantPaddingBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local numberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(numberOfDimensions, NewConstantPaddingBlock.headPaddingDimensionSizeArray, NewConstantPaddingBlock.tailPaddingDimensionSizeArray)

		if (#headPaddingDimensionSizeArray > numberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end
		
		if (#tailPaddingDimensionSizeArray > numberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end

		local transformedTensor = inputTensor
		
		local value = NewConstantPaddingBlock.value
		
		chainRuleFirstDerivativeMultiplierValue = 0

		for dimension = numberOfDimensions, 1, -1 do

			local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				local tensorHeadPaddingDimensionSizeArray = table.clone(transformedTensorDimensionSizeArray)

				tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

				local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray, value)

				transformedTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, transformedTensor, dimension)
				
				chainRuleFirstDerivativeMultiplierValue = chainRuleFirstDerivativeMultiplierValue + getTotalDimensionSize(tensorHeadPaddingDimensionSizeArray)

			end

			if (tailPaddingDimensionSize >= 1) then

				local tensorTailPaddingDimensionSizeArray = table.clone(transformedTensorDimensionSizeArray)

				tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

				local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray, value)

				transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, tailPaddingTensor, dimension)
				
				chainRuleFirstDerivativeMultiplierValue = chainRuleFirstDerivativeMultiplierValue + getTotalDimensionSize(tensorTailPaddingDimensionSizeArray)

			end

		end

		return transformedTensor

	end)

	NewConstantPaddingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local numberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray = padArraysToEqualLengths(numberOfDimensions, NewConstantPaddingBlock.headPaddingDimensionSizeArray, NewConstantPaddingBlock.tailPaddingDimensionSizeArray)

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)
		
		local value = NewConstantPaddingBlock.value

		for dimension = 1, numberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(initialPartialFirstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)
		
		if (value ~= 0) then
			
			chainRuleFirstDerivativeTensor = AqwamTensorLibrary:applyFunction(chainRuleFirstDerivativeMultiplierFunction, chainRuleFirstDerivativeTensor, inputTensor, value, chainRuleFirstDerivativeMultiplierValue)
			
		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewConstantPaddingBlock

end

return ConstantPaddingBlock