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

local Convolution1DBlock = {}

Convolution1DBlock.__index = Convolution1DBlock

setmetatable(Convolution1DBlock, BaseConvolutionBlock)

local defaultNumberOfKernels = 2

local defaultKernelDimensionSize = 2

local defaultStrideDimensionSize = 1

function Convolution1DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local channelSize = parameterDictionary.channelSize

	local numberOfKernels = parameterDictionary.numberOfKernels or defaultNumberOfKernels

	local kernelDimensionSize = parameterDictionary.kernelDimensionSize or defaultKernelDimensionSize

	local strideDimensionSize = parameterDictionary.strideDimensionSize or defaultStrideDimensionSize

	if (not channelSize) then error("No channel size.") end

	if (type(kernelDimensionSize) ~= "number") then error("The kernel dimension size must be a number.") end

	if (type(strideDimensionSize) ~= "number") then error("The stride dimension size must be a number.") end
	
	parameterDictionary.dimensionSizeArray = {numberOfKernels, channelSize, kernelDimensionSize}

	local NewConvolution1DBlock = BaseConvolutionBlock.new(parameterDictionary)

	setmetatable(NewConvolution1DBlock, Convolution1DBlock)

	NewConvolution1DBlock:setName("Convolution1D")

	NewConvolution1DBlock:setFirstDerivativeFunctionRequiresTransformedTensor(true)

	NewConvolution1DBlock.channelSize = channelSize

	NewConvolution1DBlock.numberOfKernels = numberOfKernels

	NewConvolution1DBlock.kernelDimensionSize = kernelDimensionSize

	NewConvolution1DBlock.strideDimensionSize = strideDimensionSize

	NewConvolution1DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial convolution function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end
		
		if (inputTensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

		local strideDimensionSize = NewConvolution1DBlock.strideDimensionSize

		local dimensionSizeArray = NewConvolution1DBlock.dimensionSizeArray

		local transformedTensorDimensionSizeArray = {inputTensorDimensionSizeArray[1], NewConvolution1DBlock.numberOfKernels}

		local inputDimensionSize = inputTensorDimensionSizeArray[3]

		local outputDimensionSize = ((inputDimensionSize - dimensionSizeArray[3]) / strideDimensionSize) + 1

		transformedTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

		local transformedTensor = {}

		local coroutineArray = {}

		local weight3DTensor = NewConvolution1DBlock:getWeightTensor(true)

		if (not weight3DTensor) then

			weight3DTensor = NewConvolution1DBlock:generateWeightTensor()

			NewConvolution1DBlock:setWeightTensor(weight3DTensor, true)

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

	NewConvolution1DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

		local kernelDimensionSize = NewConvolution1DBlock.kernelDimensionSize

		local strideDimensionSize = NewConvolution1DBlock.strideDimensionSize

		local weight3DTensor = NewConvolution1DBlock:getWeightTensor(true)

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

	Convolution1DBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]
		
		local initialPartialFirstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local kernelDimensionSize = NewConvolution1DBlock.kernelDimensionSize

		local strideDimensionSize = NewConvolution1DBlock.strideDimensionSize

		local dimensionSizeArray = NewConvolution1DBlock.dimensionSizeArray

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
	
	return NewConvolution1DBlock

end

return Convolution1DBlock
