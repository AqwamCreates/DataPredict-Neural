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

local BaseActivationBlock = require(script.Parent.BaseActivationBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

StableSoftmaxBlock = {}

StableSoftmaxBlock.__index = StableSoftmaxBlock

setmetatable(StableSoftmaxBlock, BaseActivationBlock)

local function sumToChainRuleFirstDerivativeTensorWhenSameDimensionIndex(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, targetTensor)

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			sumToChainRuleFirstDerivativeTensorWhenSameDimensionIndex(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

		end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local value = tensor[i]

			targetTensor[i] = targetTensor[i] + (value * (1 - value))

		end

	end

end

local function sumToChainRuleFirstDerivativeTensorWhenDifferentDimensionIndex(tensor1, tensor2, dimensionSizeArray, numberOfDimensions, currentDimension, targetTensor)

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			sumToChainRuleFirstDerivativeTensorWhenDifferentDimensionIndex(tensor1[i], tensor2[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

		end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local calculatedTensor = tensor1[i] * (1 - tensor2[i])

			targetTensor[i] = targetTensor[i] + calculatedTensor

		end

	end

end

local function calculateChainRuleFirstDerivativeTensor(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, targetTensor, sumDimension)

	local nextDimension = currentDimension + 1

	if (currentDimension < (sumDimension - 1)) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			calculateChainRuleFirstDerivativeTensor(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i], sumDimension)

		end

	else

		for i, subTensor1 in ipairs(tensor) do

			for j, subTensor2 in ipairs(tensor) do

				if (i == j) then

					sumToChainRuleFirstDerivativeTensorWhenSameDimensionIndex(subTensor1, dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

				else

					sumToChainRuleFirstDerivativeTensorWhenDifferentDimensionIndex(subTensor1, subTensor2, dimensionSizeArray, numberOfDimensions, nextDimension, targetTensor[i])

				end

			end

		end

	end	

end

--[[

local function sumToChainRuleFirstDerivativeTensor(tensor, numberOfDimensions, currentDimension, dimensionSizeArray, currentDimensionIndexArray1, currentDimensionIndexArray2, targetTensor, hasSameDimensionIndex)
	
	if (currentDimension < numberOfDimensions) then
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do
			
			currentDimensionIndexArray1[currentDimension] = i

			currentDimensionIndexArray2[currentDimension] = i

			sumToChainRuleFirstDerivativeTensor(tensor, numberOfDimensions, currentDimension + 1, dimensionSizeArray, currentDimensionIndexArray1, currentDimensionIndexArray2, targetTensor, hasSameDimensionIndex)
			
		end

	else
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do
			
			currentDimensionIndexArray1[currentDimension] = i

			currentDimensionIndexArray2[currentDimension] = i
			
			print(currentDimensionIndexArray2)
			
			print(tensor)
			
			local currentTargetValue = AqwamTensorLibrary:getValue(tensor, currentDimensionIndexArray1)
			
			local firstValue = AqwamTensorLibrary:getValue(tensor, currentDimensionIndexArray1)
			
			local secondValue = AqwamTensorLibrary:getValue(tensor, currentDimensionIndexArray2)
			
			local calculatedValue
			
			if (hasSameDimensionIndex) then
				
				calculatedValue = firstValue * (1 - secondValue)
				
			else
				
				calculatedValue = -firstValue * secondValue
				
			end
			
			local newTargetValue = currentTargetValue + calculatedValue
			
			AqwamTensorLibrary:setValue(targetTensor, newTargetValue, currentDimensionIndexArray1)
			
		end
		
	end
	
end

local function calculateChainRuleFirstDerivativeTensor(tensor, numberOfDimensions, currentDimension, dimensionSizeArray, currentDimensionIndexArray, targetTensor, sumDimension)
	
	if (currentDimension ~= sumDimension) then
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do
			
			currentDimensionIndexArray[currentDimension] = i
			
			calculateChainRuleFirstDerivativeTensor(tensor, numberOfDimensions, currentDimension + 1, dimensionSizeArray, currentDimensionIndexArray, targetTensor, sumDimension)
			
		end

	elseif (currentDimension == sumDimension) then
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do
			
			currentDimensionIndexArray[currentDimension] = i

			local copyOfCurrentDimensionIndexArray = table.clone(currentDimensionIndexArray)
			
			for j = 1, dimensionSizeArray[currentDimension], 1 do
				
				copyOfCurrentDimensionIndexArray[currentDimension] = j

				sumToChainRuleFirstDerivativeTensor(tensor, numberOfDimensions, currentDimension, dimensionSizeArray, currentDimensionIndexArray, copyOfCurrentDimensionIndexArray, targetTensor, (i == j))

			end
			
		end

	end	
	
end

--]]

function StableSoftmaxBlock.new(parameterDictionary)

	local NewStableSoftmaxBlock = BaseActivationBlock.new()

	setmetatable(NewStableSoftmaxBlock, StableSoftmaxBlock)

	NewStableSoftmaxBlock:setName("StableSoftmax")

	NewStableSoftmaxBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(true)

	parameterDictionary = parameterDictionary or {}

	NewStableSoftmaxBlock.dimension = parameterDictionary.dimension or 1

	NewStableSoftmaxBlock:setFunction(function(inputTensorArray)
		
		local inputTensor = inputTensorArray[1]
		
		local maximumValue = AqwamTensorLibrary:findMaximumValue(inputTensor)
		
		local subtractedZTensor = AqwamTensorLibrary:subtract(inputTensor, maximumValue)

		local exponentInputTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

		local summedExponentInputTensor = AqwamTensorLibrary:sum(exponentInputTensor, NewStableSoftmaxBlock.dimension)

		return AqwamTensorLibrary:divide(exponentInputTensor, summedExponentInputTensor)

	end)

	NewStableSoftmaxBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local partialChainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray)
		
		calculateChainRuleFirstDerivativeTensor(transformedTensor, dimensionSizeArray, #dimensionSizeArray, 1, partialChainRuleFirstDerivativeTensor, NewStableSoftmaxBlock.dimension)
		
		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, partialChainRuleFirstDerivativeTensor)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewStableSoftmaxBlock

end

function StableSoftmaxBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimension = parameterDictionary.dimension or self.dimension

end

return StableSoftmaxBlock