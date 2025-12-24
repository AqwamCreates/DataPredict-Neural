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

local BaseConvolutionBlock = require(script.Parent.BaseConvolutionBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticConvolution1DBlock = {}

AutomaticConvolution1DBlock.__index = AutomaticConvolution1DBlock

setmetatable(AutomaticConvolution1DBlock, BaseConvolutionBlock)

local defaultNumberOfKernels = 2

local defaultKernelDimensionSize = 2

local defaultStrideDimensionSize = 1

function AutomaticConvolution1DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAutomaticConvolution1DBlock = BaseConvolutionBlock.new(parameterDictionary)

	setmetatable(NewAutomaticConvolution1DBlock, AutomaticConvolution1DBlock)

	NewAutomaticConvolution1DBlock:setName("AutomaticConvolution1D")

	local numberOfKernels = parameterDictionary.numberOfKernels or defaultNumberOfKernels

	local kernelDimensionSize = parameterDictionary.kernelDimensionSize or defaultKernelDimensionSize

	local strideDimensionSize = parameterDictionary.strideDimensionSize or defaultStrideDimensionSize

	NewAutomaticConvolution1DBlock.numberOfKernels = numberOfKernels

	NewAutomaticConvolution1DBlock.kernelDimensionSize = kernelDimensionSize

	NewAutomaticConvolution1DBlock.strideDimensionSize = strideDimensionSize

	NewAutomaticConvolution1DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial convolution function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSize = NewAutomaticConvolution1DBlock.kernelDimensionSize

		local transformedTensorDimensionSizeArray = {inputTensorDimensionSizeArray[1], NewAutomaticConvolution1DBlock.numberOfKernels}

		local inputDimensionSize = inputTensorDimensionSizeArray[3]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSize) / strideDimensionSize) + 1

		transformedTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

		local transformedTensor = {}

		local coroutineArray = {}

		local weight3DTensor = NewAutomaticConvolution1DBlock:getWeightTensor(true)

		local dimensionSizeArray = NewAutomaticConvolution1DBlock.dimensionSizeArray

		if (not weight3DTensor) then

			dimensionSizeArray = {NewAutomaticConvolution1DBlock.numberOfKernels, inputTensorDimensionSizeArray[2], kernelDimensionSize}

			NewAutomaticConvolution1DBlock.dimensionSizeArray = dimensionSizeArray

			weight3DTensor = NewAutomaticConvolution1DBlock:generateWeightTensor(dimensionSizeArray)

			NewAutomaticConvolution1DBlock:setWeightTensor(weight3DTensor, true)

		elseif (not dimensionSizeArray) then

			dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weight3DTensor)

		end

		for a = 1, transformedTensorDimensionSizeArray[1], 1 do

			local subInputTensor = inputTensor[a]

			transformedTensor[a] = {}

			for w, weight2DTensor in ipairs(weight3DTensor) do

				transformedTensor[a][w] = {}

				BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function() -- Too slow. I had to use coroutines to speed it up.

					for c = 1, transformedTensorDimensionSizeArray[3], 1 do

						local originDimensionIndexArray = {1, (c - 1) * strideDimensionSize + 1}

						local targetDimensionIndexArray = {dimensionSizeArray[2], (c - 1) * strideDimensionSize + kernelDimensionSize}

						local extractedInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

						local subZTensor = AqwamTensorLibrary:multiply(extractedInputTensor, weight2DTensor)

						transformedTensor[a][w][c] = AqwamTensorLibrary:sum(subZTensor)

					end

				end)

			end

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return transformedTensor

	end)

	NewAutomaticConvolution1DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

		local kernelDimensionSize = NewAutomaticConvolution1DBlock.kernelDimensionSize

		local strideDimensionSize = NewAutomaticConvolution1DBlock.strideDimensionSize

		local weight3DTensor = NewAutomaticConvolution1DBlock:getWeightTensor(true)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local numberOfInputChannels = inputTensorDimensionSizeArray[2]

		local coroutineArray = {}

		for kernelIndex, weight2DTensor in ipairs(weight3DTensor) do -- So many for loops.

			BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function() -- The calculation here is so slow that I am forced to use coroutines here. What the fuck.

				for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

					for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

						local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][kernelIndex][c]

						local originDimensionIndex = ((c - 1) * strideDimensionSize)

						--[[
							
							Since the target dimension index array can be determined by adding the kernel dimension size array to the origin index array, we don't need to find the target dimension index array. 
							Also we don't need to add 1 since the the kernel dimension size array "for" loop iteration will start with 1 and add to the origin dimension index array.
							
						--]]

						local subPartialFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeValue, weight2DTensor)

						for inputChannelIndex = 1, numberOfInputChannels, 1 do

							for i = 1, kernelDimensionSize, 1 do

								chainRuleFirstDerivativeTensor[a][inputChannelIndex][originDimensionIndex + i] = chainRuleFirstDerivativeTensor[a][inputChannelIndex][originDimensionIndex + i] + subPartialFirstDerivativeTensor[inputChannelIndex][i]

							end

						end

					end

				end

			end)

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	NewAutomaticConvolution1DBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]
		
		local initialPartialFirstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local kernelDimensionSize = NewAutomaticConvolution1DBlock.kernelDimensionSize

		local strideDimensionSize = NewAutomaticConvolution1DBlock.strideDimensionSize

		local dimensionSizeArray = NewAutomaticConvolution1DBlock.dimensionSizeArray

		local firstDerivativeTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray)

		local coroutineArray = {}

		for kernelIndex = 1, dimensionSizeArray[1], 1 do

			for kernelChannelIndex = 1, dimensionSizeArray[2], 1 do

				BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function()

					for a = 1, initialPartialFirstDerivativeTensorDimensionSizeArray[1], 1 do

						local subInputTensor = inputTensor[a][kernelChannelIndex]
					
						local subInitialPartialFirstDerivativeTensor = initialPartialFirstDerivativeTensor[a][kernelChannelIndex]

						for c = 1, initialPartialFirstDerivativeTensorDimensionSizeArray[3], 1 do

							local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

							local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

							local extractedSubInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

							local subNewFirstDerivativeTensor = AqwamTensorLibrary:multiply(extractedSubInputTensor, subInitialPartialFirstDerivativeTensor[c])

							firstDerivativeTensor[kernelIndex][kernelChannelIndex] = AqwamTensorLibrary:add(firstDerivativeTensor[kernelIndex][kernelChannelIndex], subNewFirstDerivativeTensor)

						end

					end

				end)

			end

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return {firstDerivativeTensor}

	end)

	return NewAutomaticConvolution1DBlock

end

return AutomaticConvolution1DBlock
