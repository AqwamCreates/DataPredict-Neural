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

AveragePooling2DBlock = {}

AveragePooling2DBlock.__index = AveragePooling2DBlock

setmetatable(AveragePooling2DBlock, BasePoolingBlock)

local defaultKernelDimensionSizeArray = {2, 2}

local defaultStrideDimensionSizeArray = {1, 1}

function AveragePooling2DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAveragePooling2DBlock = BasePoolingBlock.new()

	setmetatable(NewAveragePooling2DBlock, AveragePooling2DBlock)

	NewAveragePooling2DBlock:setName("AveragePooling2D")

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or defaultKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or defaultStrideDimensionSizeArray

	if (#kernelDimensionSizeArray ~= 2) then error("The number of dimensions for the kernel dimension size array does not equal to 2.") end

	if (#strideDimensionSizeArray ~= 2) then error("The number of dimensions for the stride dimension size array does not equal to 2.") end

	NewAveragePooling2DBlock.kernelDimensionSizeArray = kernelDimensionSizeArray

	NewAveragePooling2DBlock.strideDimensionSizeArray = strideDimensionSizeArray

	NewAveragePooling2DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial average pooling function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSizeArray = NewAveragePooling2DBlock.kernelDimensionSizeArray 

		local strideDimensionSizeArray = NewAveragePooling2DBlock.strideDimensionSizeArray

		local transformedTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

		for dimension = 1, 2, 1 do

			local inputDimensionSize = inputTensorDimensionSizeArray[dimension + 2]

			local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

			transformedTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

		end

		local transformedTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray)

		for a = 1, transformedTensorDimensionSizeArray[1], 1 do

			for b = 1, transformedTensorDimensionSizeArray[2], 1 do

				for c = 1, transformedTensorDimensionSizeArray[3], 1 do

					for d = 1, transformedTensorDimensionSizeArray[4], 1 do

						local subInputTensor = inputTensor[a][b]

						local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2]}

						local extractedInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

						local averageValue = AqwamTensorLibrary:mean(extractedInputTensor)

						transformedTensor[a][b][c][d] = averageValue

					end

				end

			end

		end

		return transformedTensor

	end)

	NewAveragePooling2DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSizeArray = NewAveragePooling2DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewAveragePooling2DBlock.strideDimensionSizeArray

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensorArray[1])

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		local kernelArea = kernelDimensionSizeArray[3] * kernelDimensionSizeArray[4]

		for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

			for b = 1, initialPartialFirstDerivativeTensorSizeArray[2], 1 do

				for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

					for d = 1, initialPartialFirstDerivativeTensorSizeArray[4], 1 do

						local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][b][c][d]

						local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2]}

						for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

							for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

								chainRuleFirstDerivativeTensor[a][b][x][y] = chainRuleFirstDerivativeTensor[a][b][x][y] + initialPartialFirstDerivativeValue

							end

						end

					end

				end

			end

		end

		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(chainRuleFirstDerivativeTensor, kernelArea)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewAveragePooling2DBlock

end

return AveragePooling2DBlock