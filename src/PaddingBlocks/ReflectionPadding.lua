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

local ReflectionPaddingBlock = {}

ReflectionPaddingBlock.__index = ReflectionPaddingBlock

setmetatable(ReflectionPaddingBlock, BasePaddingBlock)

local defaultHeadPaddingDimensionSizeArray = {1, 1}

local defaultTailPaddingDimensionSizeArray = {1, 1}

local function padArraysToEqualLengths(numberOfDimensions, headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray)

	local headPaddingNumberOfDimensionsOffset = numberOfDimensions - #headPaddingDimensionSizeArray

	local tailPaddingNumberOfDimensionsOffset = numberOfDimensions - #tailPaddingDimensionSizeArray 

	if (headPaddingNumberOfDimensionsOffset ~= 0) then for i = 1, headPaddingNumberOfDimensionsOffset, 1 do table.insert(headPaddingDimensionSizeArray, 1, 0) end end

	if (tailPaddingNumberOfDimensionsOffset ~= 0) then for i = 1, tailPaddingNumberOfDimensionsOffset, 1 do table.insert(tailPaddingDimensionSizeArray, 1, 0) end end

	return headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray

end

local function incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	for i = #dimensionIndexArray, 1, -1 do

		dimensionIndexArray[i] = dimensionIndexArray[i] + 1

		if (dimensionIndexArray[i] <= dimensionSizeArray[i]) then break end

		dimensionIndexArray[i] = 1

	end

	return dimensionIndexArray

end

local function checkIfDimensionIndexArraysAreEqual(dimensionIndexArray1, dimensionIndexArray2)

	if (#dimensionIndexArray1 ~= #dimensionIndexArray2) then return false end

	for i, index in ipairs(dimensionIndexArray1) do

		if (index ~= dimensionIndexArray2[i]) then return false end

	end

	return true

end

function ReflectionPaddingBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewReflectionPaddingBlock = BasePaddingBlock.new()

	setmetatable(NewReflectionPaddingBlock, ReflectionPaddingBlock)

	NewReflectionPaddingBlock:setName("ReflectionPadding")

	NewReflectionPaddingBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewReflectionPaddingBlock.headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or defaultHeadPaddingDimensionSizeArray

	NewReflectionPaddingBlock.tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or defaultTailPaddingDimensionSizeArray

	NewReflectionPaddingBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local numberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(numberOfDimensions, NewReflectionPaddingBlock.headPaddingDimensionSizeArray, NewReflectionPaddingBlock.tailPaddingDimensionSizeArray)

		if (#headPaddingDimensionSizeArray > numberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end
		
		if (#tailPaddingDimensionSizeArray > numberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end
		
		for dimension = 1, numberOfDimensions, 1 do
			
			local inputTensorDimensionSize = inputTensorDimensionSizeArray[dimension]
			
			local headDimensionSize = headPaddingDimensionSizeArray[dimension]
			
			local tailDimensionSize = tailPaddingDimensionSizeArray[dimension]
			
			local errorStringEnding = " must not be greater or equal to the dimension size of " .. inputTensorDimensionSize .. " from the input tensor."
			
			if (headDimensionSize >= inputTensorDimensionSize) then error("The head padding dimension size of " .. headDimensionSize .. " at dimension " .. dimension .. errorStringEnding) end
			
			if (tailDimensionSize >= inputTensorDimensionSize) then error("The tail padding dimension size of " .. tailDimensionSize .. " at dimension " .. dimension .. errorStringEnding) end
			
		end

		local transformedTensor = inputTensor

		for dimension = numberOfDimensions, 1, -1 do

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

				local transformedTensorDimensionSize = transformedTensorDimensionSizeArray[dimension]

				local transformedTensorStartDimensionIndexArray = table.create(numberOfDimensions, 1)

				local transformedTensorEndDimensionIndexArray = table.clone(transformedTensorDimensionSizeArray)
				
				local startingIndex = 1

				for i = 1, headPaddingDimensionSize, 1 do
					
					local currentIndex = startingIndex + i 
					
					transformedTensorStartDimensionIndexArray[dimension] = currentIndex

					transformedTensorEndDimensionIndexArray[dimension] = currentIndex
					
					startingIndex = startingIndex + 1
					
					local extractedInputTensor = AqwamTensorLibrary:extract(transformedTensor, transformedTensorStartDimensionIndexArray, transformedTensorEndDimensionIndexArray)
					
					transformedTensor = AqwamTensorLibrary:concatenate(extractedInputTensor, transformedTensor, dimension) 
					
				end

			end

			if (tailPaddingDimensionSize >= 1) then

				local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

				local transformedTensorDimensionSize = transformedTensorDimensionSizeArray[dimension]

				local transformedTensorStartDimensionIndexArray = table.create(numberOfDimensions, 1)

				local transformedTensorEndDimensionIndexArray = table.clone(transformedTensorDimensionSizeArray)
				
				local startingIndex = transformedTensorDimensionSize
				
				for i = 1, tailPaddingDimensionSize, 1 do
					
					local currentIndex = startingIndex - i
					
					transformedTensorStartDimensionIndexArray[dimension] = currentIndex

					transformedTensorEndDimensionIndexArray[dimension] = currentIndex
					
					local extractedInputTensor = AqwamTensorLibrary:extract(transformedTensor, transformedTensorStartDimensionIndexArray, transformedTensorEndDimensionIndexArray)
					
					transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, extractedInputTensor, dimension) 
					
				end

			end

		end

		return transformedTensor

	end)

	NewReflectionPaddingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local inputTensorNumberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(inputTensorNumberOfDimensions, NewReflectionPaddingBlock.headPaddingDimensionSizeArray, NewReflectionPaddingBlock.tailPaddingDimensionSizeArray)

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)

		for dimension = 1, inputTensorNumberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(initialPartialFirstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)
		
		local chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)
		
		local originExtractionDimensionIndexArray = table.create(inputTensorNumberOfDimensions, 1)
		
		for dimension = 1, inputTensorNumberOfDimensions, 1 do -- Gradient edge cases.
			
			local inputTensorDimensionSize = inputTensorDimensionSizeArray[dimension]

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]
			
			if (headPaddingDimensionSize >= 1) then
				
				if (inputTensorDimensionSize > 1) then
					
					local remainingExtractionDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)

					remainingExtractionDimensionIndexArray[dimension] = 1

					local extractedChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, originExtractionDimensionIndexArray, remainingExtractionDimensionIndexArray)

					local targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)

					targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] = targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] + 1

					local targetChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, targetChainRuleFirstDerivativeTensorHeadDimensionIndexArray, inputTensorDimensionSizeArray)
					
					targetChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:add(targetChainRuleFirstDerivativeHeadTensor, 1)

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:concatenate(extractedChainRuleFirstDerivativeHeadTensor, targetChainRuleFirstDerivativeHeadTensor, dimension)	
					
				else
					
					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:add(chainRuleFirstDerivativeMultiplierTensor, 1)
					
				end
				
			end
			
			if (tailPaddingDimensionSize >= 1) then -- Tail gradient edge cases.

				if (inputTensorDimensionSize > 1) then

					local remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)

					remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] - 1

					local extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)

					extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = inputTensorDimensionSizeArray[dimension]

					local targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, originExtractionDimensionIndexArray, remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray)

					local remainingChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeMultiplierTensor, extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray, inputTensorDimensionSizeArray)

					targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:add(targetChainRuleFirstDerivativeTailTensor, 1)

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:concatenate(targetChainRuleFirstDerivativeTailTensor, remainingChainRuleFirstDerivativeTailTensor, dimension)

				else

					chainRuleFirstDerivativeMultiplierTensor = AqwamTensorLibrary:add(chainRuleFirstDerivativeMultiplierTensor, tailPaddingDimensionSize)

				end

			end
			
		end
		
		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensor, chainRuleFirstDerivativeMultiplierTensor)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewReflectionPaddingBlock

end

return ReflectionPaddingBlock
