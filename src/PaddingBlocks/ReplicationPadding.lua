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

ReplicationPaddingBlock = {}

ReplicationPaddingBlock.__index = ReplicationPaddingBlock

setmetatable(ReplicationPaddingBlock, BasePaddingBlock)

local defaultHeadPaddingDimensionSizeArray = {1, 1}

local defaultTailPaddingDimensionSizeArray = {1, 1}

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

local function bruteForceCornerLocationsFromDimensionSizeArray(cornerLocationDimensionIndexArrayArray, dimensionSizeArray, currentDimension, numberOfDimensions, dimensionIndexArray)
	
	if (currentDimension <= numberOfDimensions) then
		
		local newCurrentDimension = currentDimension + 1
		
		dimensionIndexArray[currentDimension] = 1

		bruteForceCornerLocationsFromDimensionSizeArray(cornerLocationDimensionIndexArrayArray, dimensionSizeArray, newCurrentDimension, numberOfDimensions, dimensionIndexArray)

		dimensionIndexArray[currentDimension] = dimensionSizeArray[currentDimension]

		bruteForceCornerLocationsFromDimensionSizeArray(cornerLocationDimensionIndexArrayArray, dimensionSizeArray, newCurrentDimension, numberOfDimensions, dimensionIndexArray)
		
	else
		
		table.insert(cornerLocationDimensionIndexArrayArray, table.clone(dimensionIndexArray))

	end
	
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

function ReplicationPaddingBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewReplicationPaddingBlock = BasePaddingBlock.new()

	setmetatable(NewReplicationPaddingBlock, ReplicationPaddingBlock)

	NewReplicationPaddingBlock:setName("ReplicationPadding")

	NewReplicationPaddingBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	NewReplicationPaddingBlock.headPaddingDimensionSizeArray = parameterDictionary.headPaddingDimensionSizeArray or defaultHeadPaddingDimensionSizeArray

	NewReplicationPaddingBlock.tailPaddingDimensionSizeArray = parameterDictionary.tailPaddingDimensionSizeArray or defaultTailPaddingDimensionSizeArray

	NewReplicationPaddingBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local numberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(numberOfDimensions, NewReplicationPaddingBlock.headPaddingDimensionSizeArray, NewReplicationPaddingBlock.tailPaddingDimensionSizeArray)

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
				
				transformedTensorStartDimensionIndexArray[dimension] = 1

				transformedTensorEndDimensionIndexArray[dimension] = 1
				
				local extractedInputTensor = AqwamTensorLibrary:extract(transformedTensor, transformedTensorStartDimensionIndexArray, transformedTensorEndDimensionIndexArray)

				for i = 1, headPaddingDimensionSize, 1 do transformedTensor = AqwamTensorLibrary:concatenate(extractedInputTensor, transformedTensor, dimension) end

			end

			if (tailPaddingDimensionSize >= 1) then

				local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

				local transformedTensorDimensionSize = transformedTensorDimensionSizeArray[dimension]

				local transformedTensorStartDimensionIndexArray = table.create(numberOfDimensions, 1)

				local transformedTensorEndDimensionIndexArray = table.clone(transformedTensorDimensionSizeArray)
				
				transformedTensorStartDimensionIndexArray[dimension] = transformedTensorDimensionSize

				transformedTensorEndDimensionIndexArray[dimension] = transformedTensorDimensionSize
				
				local extractedInputTensor = AqwamTensorLibrary:extract(transformedTensor, transformedTensorStartDimensionIndexArray, transformedTensorEndDimensionIndexArray)

				for i = 1, tailPaddingDimensionSize, 1 do transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, extractedInputTensor, dimension) end

			end

		end

		return transformedTensor

	end)

	NewReplicationPaddingBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local inputTensorNumberOfDimensions = #inputTensorDimensionSizeArray

		local headPaddingDimensionSizeArray, tailPaddingDimensionSizeArray = padArraysToEqualLengths(inputTensorNumberOfDimensions, NewReplicationPaddingBlock.headPaddingDimensionSizeArray, NewReplicationPaddingBlock.tailPaddingDimensionSizeArray)

		local originDimensionIndexArray = {}

		local targetDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)

		for dimension = 1, inputTensorNumberOfDimensions, 1 do

			originDimensionIndexArray[dimension] = (headPaddingDimensionSizeArray[dimension] or 0) + 1

			targetDimensionIndexArray[dimension] = targetDimensionIndexArray[dimension] + (headPaddingDimensionSizeArray[dimension] or 0)

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:extract(initialPartialFirstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)
		
		local originExtractionDimensionIndexArray = table.create(inputTensorNumberOfDimensions, 1)
		
		for dimension = 1, inputTensorNumberOfDimensions, 1 do
			
			local inputTensorDimensionSize = inputTensorDimensionSizeArray[dimension]

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then -- Head gradient edge cases.
				
				if (inputTensorDimensionSize > 1) then
					
					local targetExtractionDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)

					targetExtractionDimensionIndexArray[dimension] = 1

					local extractedChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, originExtractionDimensionIndexArray, targetExtractionDimensionIndexArray)

					extractedChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:multiply(extractedChainRuleFirstDerivativeHeadTensor, headPaddingDimensionSize)
					
					local remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)
					
					remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] = remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray[dimension] + 1

					local remainingChainRuleFirstDerivativeHeadTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, remainingChainRuleFirstDerivativeTensorHeadDimensionIndexArray, inputTensorDimensionSizeArray)
					
					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:concatenate(extractedChainRuleFirstDerivativeHeadTensor, remainingChainRuleFirstDerivativeHeadTensor, dimension)		
					
				else
					
					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensor, headPaddingDimensionSize)
					
				end

			end
			
			if (tailPaddingDimensionSize >= 1) then -- Tail gradient edge cases.
				
				if (inputTensorDimensionSize > 1) then

					local remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(inputTensorDimensionSizeArray)

					remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] - 1

					local extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray = table.clone(originExtractionDimensionIndexArray)
					
					extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray[dimension] = inputTensorDimensionSizeArray[dimension]

					local remainingChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, originExtractionDimensionIndexArray, remainingChainRuleFirstDerivativeTensorTailDimensionIndexArray)

					local targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:extract(chainRuleFirstDerivativeTensor, extractedChainRuleFirstDerivativeTensorTailDimensionIndexArray, inputTensorDimensionSizeArray)

					targetChainRuleFirstDerivativeTailTensor = AqwamTensorLibrary:multiply(targetChainRuleFirstDerivativeTailTensor, tailPaddingDimensionSize)

					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:concatenate(remainingChainRuleFirstDerivativeTailTensor, targetChainRuleFirstDerivativeTailTensor, dimension)
					
				else
					
					chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(chainRuleFirstDerivativeTensor, tailPaddingDimensionSize)
					
				end

			end

		end
		
		--[[
		
		if (inputTensorNumberOfDimensions >= 2) then -- Gradient corner cases. Apparently, the size of the corner is literally the size of tail/head padding multiplied with the adjacent tail/head padding.
			
			local cornerLocationDimensionIndexArrayArray = {}
			
			local dimensionIndexArray = table.create(inputTensorNumberOfDimensions, 1)
			
			bruteForceCornerLocationsFromDimensionSizeArray(cornerLocationDimensionIndexArrayArray, inputTensorDimensionSizeArray, 1, inputTensorNumberOfDimensions, dimensionIndexArray)
			
			local headheadAreaArray = {}
			
			local tailheadArea
			
			for _, cornerLocationDimensionIndexArray in ipairs(cornerLocationDimensionIndexArrayArray) do
				
				local chainRuleFirstDerivativeValue = AqwamTensorLibrary:getValue(chainRuleFirstDerivativeTensor, cornerLocationDimensionIndexArray)
				
				AqwamTensorLibrary:setValue(chainRuleFirstDerivativeTensor, cornerLocationDimensionIndexArray, chainRuleFirstDerivativeValue)
				
			end

		end
		
		--]]

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewReplicationPaddingBlock

end

return ReplicationPaddingBlock