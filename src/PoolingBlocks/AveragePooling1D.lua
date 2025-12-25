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

local AveragePooling1DBlock = {}

AveragePooling1DBlock.__index = AveragePooling1DBlock

setmetatable(AveragePooling1DBlock, BasePoolingBlock)

local defaultKernelDimensionSize = 2

local defaultStrideDimensionSize = 2

function AveragePooling1DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAveragePooling1DBlock = BasePoolingBlock.new()

	setmetatable(NewAveragePooling1DBlock, AveragePooling1DBlock)

	NewAveragePooling1DBlock:setName("AveragePooling1D")

	local kernelDimensionSize = parameterDictionary.kernelDimensionSize or defaultKernelDimensionSize

	local strideDimensionSize = parameterDictionary.strideDimensionSize or defaultStrideDimensionSize

	if (type(kernelDimensionSize) ~= "number") then error("The kernel dimension size must be a number.") end

	if (type(strideDimensionSize) ~= "number") then error("The stride dimension size must be a number.") end

	NewAveragePooling1DBlock.kernelDimensionSize = kernelDimensionSize

	NewAveragePooling1DBlock.strideDimensionSize = strideDimensionSize

	NewAveragePooling1DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial average pooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSize = NewAveragePooling1DBlock.kernelDimensionSize 

		local strideDimensionSize = NewAveragePooling1DBlock.strideDimensionSize

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

					local averageValue = AqwamTensorLibrary:mean(extractedInputTensor)

					transformedTensor[a][b][c] = averageValue

				end

			end

		end

		return transformedTensor

	end)

	NewAveragePooling1DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSize = NewAveragePooling1DBlock.kernelDimensionSize

		local strideDimensionSize = NewAveragePooling1DBlock.strideDimensionSize

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

			for b = 1, initialPartialFirstDerivativeTensorSizeArray[2], 1 do

				for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

					local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][b][c]

					local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

					for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

						chainRuleFirstDerivativeTensor[a][b][x] = chainRuleFirstDerivativeTensor[a][b][x] + initialPartialFirstDerivativeValue

					end

				end

			end

		end

		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(chainRuleFirstDerivativeTensor, kernelDimensionSize)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewAveragePooling1DBlock

end

return AveragePooling1DBlock
