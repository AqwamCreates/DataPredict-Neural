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

local BasePoolingBlock = require(script.Parent.BasePoolingBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

MaximumPooling1DBlock = {}

MaximumPooling1DBlock.__index = MaximumPooling1DBlock

setmetatable(MaximumPooling1DBlock, BasePoolingBlock)

local defaultKernelDimensionSize = 2

local defaultStrideDimensionSize = 2

function MaximumPooling1DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMaximumPooling1DBlock = BasePoolingBlock.new()

	setmetatable(NewMaximumPooling1DBlock, MaximumPooling1DBlock)

	NewMaximumPooling1DBlock:setName("MaximumPooling1D")

	NewMaximumPooling1DBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(true)

	local kernelDimensionSize = parameterDictionary.kernelDimensionSize or defaultKernelDimensionSize

	local strideDimensionSize = parameterDictionary.strideDimensionSize or defaultStrideDimensionSize

	if (type(kernelDimensionSize) ~= "number") then error("The kernel dimension size must be a number.") end

	if (type(strideDimensionSize) ~= "number") then error("The stride dimension size must be a number.") end

	NewMaximumPooling1DBlock.kernelDimensionSize = kernelDimensionSize

	NewMaximumPooling1DBlock.strideDimensionSize = strideDimensionSize

	NewMaximumPooling1DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial minimum pooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSize = NewMaximumPooling1DBlock.kernelDimensionSize 

		local strideDimensionSize = NewMaximumPooling1DBlock.strideDimensionSize

		local transformedTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

		local inputDimensionSize = inputTensorDimensionSizeArray[3]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSize) / strideDimensionSize) + 1

		transformedTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

		local transformedTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray)

		for a = 1, transformedTensorDimensionSizeArray[1], 1 do

			for b = 1, transformedTensorDimensionSizeArray[2], 1 do

				for c = 1, transformedTensorDimensionSizeArray[3], 1 do

					local subInputTensor = inputTensor[a][b]

					local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

					local extractedInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

					local maximumValue = AqwamTensorLibrary:findMaximumValue(extractedInputTensor)

					transformedTensor[a][b][c] = maximumValue

				end

			end

		end

		return transformedTensor

	end)

	NewMaximumPooling1DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSize = NewMaximumPooling1DBlock.kernelDimensionSize

		local strideDimensionSize = NewMaximumPooling1DBlock.strideDimensionSize

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

			for b = 1, initialPartialFirstDerivativeTensorSizeArray[2], 1 do

				for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

					local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][b][c]

					local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

					for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

						if (transformedTensor[a][b][c] == inputTensor[a][b][x]) then chainRuleFirstDerivativeTensor[a][b][x] = chainRuleFirstDerivativeTensor[a][b][x] + initialPartialFirstDerivativeValue end

					end

				end

			end

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewMaximumPooling1DBlock

end

return MaximumPooling1DBlock