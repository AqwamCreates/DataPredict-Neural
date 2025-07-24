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

CircularPaddingBlock = {}

CircularPaddingBlock.__index = CircularPaddingBlock

setmetatable(CircularPaddingBlock, BasePaddingBlock)

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

function CircularPaddingBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewCircularPaddingBlock = BasePaddingBlock.new()

	setmetatable(NewCircularPaddingBlock, CircularPaddingBlock)

	NewCircularPaddingBlock:setName("CircularPadding")

	NewCircularPaddingBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewCircularPaddingBlock.headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or defaultHeadPaddingDimensionSizeArray

	NewCircularPaddingBlock.tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or defaultTailPaddingDimensionSizeArray

	NewCircularPaddingBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local numberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(numberOfDimensions, NewCircularPaddingBlock.headPaddingDimensionSizeArray, NewCircularPaddingBlock.tailPaddingDimensionSizeArray)

		if (#headPaddingDimensionSizeArray > numberOfDimensions) then error("The number of dimensions of the head padding exceeds the number of dimensions of the input tensor.") end

		if (#tailPaddingDimensionSizeArray > numberOfDimensions) then error("The number of dimensions of the tail padding exceeds the number of dimensions of the input tensor.") end
		
		local transformedTensor = inputTensor
		
		for dimension = numberOfDimensions, 1, -1 do

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]
			
			if (headPaddingDimensionSize >= 1) then
				
				local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)
				
				local transformedTensorDimensionSize = transformedTensorDimensionSizeArray[dimension]
				
				local transformedTensorStartDimensionIndexArray = table.create(numberOfDimensions, 1)
				
				local transformedTensorEndDimensionIndexArray = table.clone(transformedTensorDimensionSizeArray)
				
				transformedTensorStartDimensionIndexArray[dimension] = transformedTensorDimensionSize

				transformedTensorEndDimensionIndexArray[dimension] = transformedTensorDimensionSize
				
				for i = 1, headPaddingDimensionSize, 1 do
					
					local extractedInputTensor = AqwamTensorLibrary:extract(transformedTensor, transformedTensorStartDimensionIndexArray, transformedTensorEndDimensionIndexArray)
					
					transformedTensor = AqwamTensorLibrary:concatenate(extractedInputTensor, transformedTensor, dimension)
					
				end

			end

			if (tailPaddingDimensionSize >= 1) then
				
				local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)
				
				local transformedTensorDimensionSize = transformedTensorDimensionSizeArray[dimension]
				
				local transformedTensorStartDimensionIndexArray = table.create(numberOfDimensions, 1)

				local transformedTensorEndDimensionIndexArray = table.clone(transformedTensorDimensionSizeArray)
				
				local currentIndex = headPaddingDimensionSize + 1

				for i = 1, tailPaddingDimensionSize, 1 do
					
					transformedTensorStartDimensionIndexArray[dimension] = currentIndex

					transformedTensorEndDimensionIndexArray[dimension] = currentIndex
					
					currentIndex = currentIndex + 1

					local extractedInputTensor = AqwamTensorLibrary:extract(transformedTensor, transformedTensorStartDimensionIndexArray, transformedTensorEndDimensionIndexArray)

					transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, extractedInputTensor, dimension)

				end

			end

		end

		return transformedTensor

	end)

	NewCircularPaddingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]
		
		local initialPartialFirstDerivativeTensorDimensionSizeArray  = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor) 
		
		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local inputTensorNumberOfDimensions = #inputTensorDimensionSizeArray
		
		local initialPartialFirstDerivativeTensorNumberOfDimensions = #initialPartialFirstDerivativeTensorDimensionSizeArray
		
		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(inputTensorNumberOfDimensions, NewCircularPaddingBlock.headPaddingDimensionSizeArray, NewCircularPaddingBlock.tailPaddingDimensionSizeArray)
		
		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray, 0)
		
		local currentInitialPartialFirstDerivativeTensorDimensionIndexArray = table.create(initialPartialFirstDerivativeTensorNumberOfDimensions, 1)
		
		currentInitialPartialFirstDerivativeTensorDimensionIndexArray[initialPartialFirstDerivativeTensorNumberOfDimensions] = 0

		local currentInputTensorDimensionIndexArray = {}
		
		for dimension, inputTensorDimensionSize in ipairs(inputTensorDimensionSizeArray) do
			
			local currentTensorDimensionIndex = headPaddingDimensionSizeArray[dimension] % inputTensorDimensionSize
			
			if (currentTensorDimensionIndex == 0) then currentTensorDimensionIndex = inputTensorDimensionSize end
			
			currentInputTensorDimensionIndexArray[dimension] = currentTensorDimensionIndex
			
		end
		
		currentInputTensorDimensionIndexArray[inputTensorNumberOfDimensions] = currentInputTensorDimensionIndexArray[inputTensorNumberOfDimensions] - 1
		
		local currentChainRuleFirstDerivativeValue
		
		local initialPartialFirstDerivativeValue
		
		local newChainRuleFirstDerivativeValue
		
		repeat
			
			currentInputTensorDimensionIndexArray = incrementDimensionIndexArray(currentInputTensorDimensionIndexArray, inputTensorDimensionSizeArray)

			currentInitialPartialFirstDerivativeTensorDimensionIndexArray = incrementDimensionIndexArray(currentInitialPartialFirstDerivativeTensorDimensionIndexArray, initialPartialFirstDerivativeTensorDimensionSizeArray)
			
			currentChainRuleFirstDerivativeValue = AqwamTensorLibrary:getValue(chainRuleFirstDerivativeTensor, currentInputTensorDimensionIndexArray)
			
			initialPartialFirstDerivativeValue = AqwamTensorLibrary:getValue(initialPartialFirstDerivativeTensor, currentInitialPartialFirstDerivativeTensorDimensionIndexArray)  
			
			newChainRuleFirstDerivativeValue = currentChainRuleFirstDerivativeValue + initialPartialFirstDerivativeValue
			
			AqwamTensorLibrary:setValue(chainRuleFirstDerivativeTensor, newChainRuleFirstDerivativeValue, currentInputTensorDimensionIndexArray)
			
		until checkIfDimensionIndexArraysAreEqual(currentInitialPartialFirstDerivativeTensorDimensionIndexArray, initialPartialFirstDerivativeTensorDimensionSizeArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewCircularPaddingBlock

end

return CircularPaddingBlock