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

local Convolution2DBlock = {}

Convolution2DBlock.__index = Convolution2DBlock

setmetatable(Convolution2DBlock, BaseConvolutionBlock)

local defaultNumberOfKernels = 2

local defaultKernelDimensionSizeArray = {2, 2}

local defaultStrideDimensionSizeArray = {1, 1}

function Convolution2DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}
	
	local channelSize = parameterDictionary.channelSize

	local numberOfKernels = parameterDictionary.numberOfKernels or defaultNumberOfKernels

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or defaultKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or defaultStrideDimensionSizeArray
	
	if (not channelSize) then error("No channel size.") end

	if (#kernelDimensionSizeArray ~= 2) then error("The number of dimensions for the kernel dimension size array does not equal to 2.") end

	if (#strideDimensionSizeArray ~= 2) then error("The number of dimensions for the stride dimension size array does not equal to 2.") end
	
	local dimensionSizeArray = table.clone(kernelDimensionSizeArray)

	table.insert(dimensionSizeArray, 1, numberOfKernels)

	table.insert(dimensionSizeArray, 2, channelSize)
	
	parameterDictionary.dimensionSizeArray = dimensionSizeArray
	
	local NewConvolution2DBlock = BaseConvolutionBlock.new(parameterDictionary)

	setmetatable(NewConvolution2DBlock, Convolution2DBlock)

	NewConvolution2DBlock:setName("Convolution2D")

	NewConvolution2DBlock.channelSize = channelSize
	
	NewConvolution2DBlock.numberOfKernels = numberOfKernels

	NewConvolution2DBlock.kernelDimensionSizeArray = kernelDimensionSizeArray

	NewConvolution2DBlock.strideDimensionSizeArray = strideDimensionSizeArray

	NewConvolution2DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial convolution function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end
		
		if (inputTensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

		local kernelDimensionSizeArray = NewConvolution2DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewConvolution2DBlock.strideDimensionSizeArray

		local dimensionSizeArray = NewConvolution2DBlock.dimensionSizeArray 

		local transformedTensorDimensionSizeArray = {inputTensorDimensionSizeArray[1], NewConvolution2DBlock.numberOfKernels}

		for dimension = 1, 2, 1 do

			local inputDimensionSize = inputTensorDimensionSizeArray[dimension + 2]

			local outputDimensionSize = ((inputDimensionSize - dimensionSizeArray[dimension + 2]) / strideDimensionSizeArray[dimension]) + 1

			transformedTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

		end

		local transformedTensor = {}

		local coroutineArray = {}

		local weight4DTensor = NewConvolution2DBlock:getWeightTensor(true)

		if (not weight4DTensor) then

			weight4DTensor = NewConvolution2DBlock:generateWeightTensor()

			NewConvolution2DBlock:setWeightTensor(weight4DTensor, true)

		end

		for a = 1, transformedTensorDimensionSizeArray[1], 1 do

			local subInputTensor = inputTensor[a]

			transformedTensor[a] = {}

			for w, weight3DTensor in ipairs(weight4DTensor) do

				transformedTensor[a][w] = {}

				BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function()

					for c = 1, transformedTensorDimensionSizeArray[3], 1 do

						transformedTensor[a][w][c] = {}

						for d = 1, transformedTensorDimensionSizeArray[4], 1 do

							local originDimensionIndexArray = {1, (c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

							local targetDimensionIndexArray = {dimensionSizeArray[2], (c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2]}

							local extractedInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

							local subZTensor = AqwamTensorLibrary:multiply(extractedInputTensor, weight3DTensor)

							transformedTensor[a][w][c][d] = AqwamTensorLibrary:sum(subZTensor)

						end

					end

				end)

			end

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return transformedTensor

	end)

	NewConvolution2DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

		local kernelDimensionSizeArray = NewConvolution2DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewConvolution2DBlock.strideDimensionSizeArray

		local weight4DTensor = NewConvolution2DBlock:getWeightTensor(true)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local numberOfInputChannels = inputTensorDimensionSizeArray[2]

		local coroutineArray = {}

		for kernelIndex, weight3DTensor in ipairs(weight4DTensor) do -- So many for loops.

			BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function() -- The calculation here is so slow that I am forced to use coroutines here. What the fuck.

				for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

					for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

						for d = 1, initialPartialFirstDerivativeTensorSizeArray[4], 1 do

							local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][kernelIndex][c][d]

							local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2]} 

							--[[
							
								Since the target dimension index array can be determined by adding the kernel dimension size array to the origin index array, we don't need to find the target dimension index array. 
								Also we don't need to add 1 since the the kernel dimension size array "for" loop iteration will start with 1 and add to the origin dimension index array.
							
							--]]

							local subPartialFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeValue, weight3DTensor)

							for inputChannelIndex = 1, numberOfInputChannels, 1 do

								for i = 1, kernelDimensionSizeArray[1], 1 do

									for j = 1, kernelDimensionSizeArray[2], 1 do

										chainRuleFirstDerivativeTensor[a][inputChannelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j] = chainRuleFirstDerivativeTensor[a][inputChannelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j] + subPartialFirstDerivativeTensor[inputChannelIndex][i][j]

									end

								end

							end

						end

					end

				end

			end)

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return {chainRuleFirstDerivativeTensor}

	end)

	NewConvolution2DBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]
		
		local initialPartialFirstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local kernelDimensionSizeArray = NewConvolution2DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewConvolution2DBlock.strideDimensionSizeArray

		local dimensionSizeArray = NewConvolution2DBlock.dimensionSizeArray

		local firstDerivativeTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray)

		local coroutineArray = {}

		for kernelIndex = 1, dimensionSizeArray[1], 1 do

			for kernelChannelIndex = 1, dimensionSizeArray[2], 1 do

				BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function()

					for a = 1, initialPartialFirstDerivativeTensorDimensionSizeArray[1], 1 do

						local subInputTensor = inputTensor[a][kernelChannelIndex]

						local subInitialPartialFirstDerivativeTensor = initialPartialFirstDerivativeTensor[a][kernelChannelIndex]

						for c = 1, initialPartialFirstDerivativeTensorDimensionSizeArray[3], 1 do
							
							for d = 1, initialPartialFirstDerivativeTensorDimensionSizeArray[4], 1 do
								
								local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

								local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2]}

								local extractedSubInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

								local subNewFirstDerivativeTensor = AqwamTensorLibrary:multiply(extractedSubInputTensor, subInitialPartialFirstDerivativeTensor[c][d])

								firstDerivativeTensor[kernelIndex][kernelChannelIndex] = AqwamTensorLibrary:add(firstDerivativeTensor[kernelIndex][kernelChannelIndex], subNewFirstDerivativeTensor)
								
							end

						end

					end

				end)

			end

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return {firstDerivativeTensor}

	end)

	return NewConvolution2DBlock

end

return Convolution2DBlock
