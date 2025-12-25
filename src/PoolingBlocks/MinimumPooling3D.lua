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

local MinimumPooling3DBlock = {}

MinimumPooling3DBlock.__index = MinimumPooling3DBlock

setmetatable(MinimumPooling3DBlock, BasePoolingBlock)

local defaultKernelDimensionSizeArray = {2, 2, 2}

local defaultStrideDimensionSizeArray = {1, 1, 1}

function MinimumPooling3DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMinimumPooling3DBlock = BasePoolingBlock.new()

	setmetatable(NewMinimumPooling3DBlock, MinimumPooling3DBlock)

	NewMinimumPooling3DBlock:setName("MinimumPooling3D")

	NewMinimumPooling3DBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(true)

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or defaultKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or defaultStrideDimensionSizeArray

	if (#kernelDimensionSizeArray ~= 3) then error("The number of dimensions for the kernel dimension size array does not equal to 3.") end

	if (#strideDimensionSizeArray ~= 3) then error("The number of dimensions for the stride dimension size array does not equal to 3.") end

	NewMinimumPooling3DBlock.kernelDimensionSizeArray = kernelDimensionSizeArray

	NewMinimumPooling3DBlock.strideDimensionSizeArray = strideDimensionSizeArray

	NewMinimumPooling3DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial minimum pooling function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSizeArray = NewMinimumPooling3DBlock.kernelDimensionSizeArray 

		local strideDimensionSizeArray = NewMinimumPooling3DBlock.strideDimensionSizeArray

		local transformedTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

		for dimension = 1, 3, 1 do

			local inputDimensionSize = inputTensorDimensionSizeArray[dimension + 2]

			local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

			transformedTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

		end

		local transformedTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray)

		for a = 1, transformedTensorDimensionSizeArray[1], 1 do

			for b = 1, transformedTensorDimensionSizeArray[2], 1 do

				for c = 1, transformedTensorDimensionSizeArray[3], 1 do

					for d = 1, transformedTensorDimensionSizeArray[4], 1 do

						for e = 1, transformedTensorDimensionSizeArray[5], 1 do

							local subInputTensor = inputTensor[a][b]

							local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

							local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2], (e - 1) * strideDimensionSizeArray[3] + kernelDimensionSizeArray[3]}

							local extractedInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

							local minimumValue = AqwamTensorLibrary:findMinimumValue(extractedInputTensor)

							transformedTensor[a][b][c][d][e] = minimumValue

						end

					end

				end

			end

		end

		return transformedTensor

	end)

	NewMinimumPooling3DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSizeArray = NewMinimumPooling3DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewMinimumPooling3DBlock.strideDimensionSizeArray

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

			for b = 1, initialPartialFirstDerivativeTensorSizeArray[2], 1 do

				for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

					for d = 1, initialPartialFirstDerivativeTensorSizeArray[4], 1 do

						for e = 1, initialPartialFirstDerivativeTensorSizeArray[5], 1 do

							local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][b][c][d][e]

							local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

							local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2], (e - 1) * strideDimensionSizeArray[3] + kernelDimensionSizeArray[3]}

							for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

								for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

									for z = originDimensionIndexArray[3], targetDimensionIndexArray[3], 1 do

										if (transformedTensor[a][b][c][d][e] == inputTensor[a][b][x][y][z]) then chainRuleFirstDerivativeTensor[a][b][x][y][z] = chainRuleFirstDerivativeTensor[a][b][x][y][z] + initialPartialFirstDerivativeValue end

									end

								end

							end

						end

					end

				end

			end

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewMinimumPooling3DBlock

end

return MinimumPooling3DBlock
