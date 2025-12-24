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

local AutomaticConvolution3DBlock = {}

AutomaticConvolution3DBlock.__index = AutomaticConvolution3DBlock

setmetatable(AutomaticConvolution3DBlock, BaseConvolutionBlock)

local defaultNumberOfKernels = 2

local defaultKernelDimensionSizeArray = {2, 2, 2}

local defaultStrideDimensionSizeArray = {1, 1, 1}

function AutomaticConvolution3DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAutomaticConvolution3DBlock = BaseConvolutionBlock.new(parameterDictionary)

	setmetatable(NewAutomaticConvolution3DBlock, AutomaticConvolution3DBlock)

	NewAutomaticConvolution3DBlock:setName("AutomaticConvolution3D")

	local numberOfKernels = parameterDictionary.numberOfKernels or defaultNumberOfKernels

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or defaultKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or defaultStrideDimensionSizeArray

	if (#kernelDimensionSizeArray ~= 3) then error("The number of dimensions for the kernel dimension size array does not equal to 3.") end

	if (#strideDimensionSizeArray ~= 3) then error("The number of dimensions for the stride dimension size array does not equal to 3.") end

	NewAutomaticConvolution3DBlock.numberOfKernels = numberOfKernels

	NewAutomaticConvolution3DBlock.kernelDimensionSizeArray = kernelDimensionSizeArray

	NewAutomaticConvolution3DBlock.strideDimensionSizeArray = strideDimensionSizeArray

	NewAutomaticConvolution3DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial convolution function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSizeArray = NewAutomaticConvolution3DBlock.kernelDimensionSizeArray 

		local transformedTensorDimensionSizeArray = {inputTensorDimensionSizeArray[1], NewAutomaticConvolution3DBlock.numberOfKernels}
		
		for dimension = 1, 3, 1 do

			local inputDimensionSize = inputTensorDimensionSizeArray[dimension + 2]

			local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

			transformedTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

		end

		local transformedTensor = {}

		local coroutineArray = {}

		local numberOfChannels = inputTensorDimensionSizeArray[2]

		local weight5DTensor = NewAutomaticConvolution3DBlock:getWeightTensor(true)

		local dimensionSizeArray = NewAutomaticConvolution3DBlock.dimensionSizeArray

		if (not weight5DTensor) then

			dimensionSizeArray = {NewAutomaticConvolution3DBlock.numberOfKernels, inputTensorDimensionSizeArray[2], kernelDimensionSizeArray[1], kernelDimensionSizeArray[2], kernelDimensionSizeArray[3]}

			NewAutomaticConvolution3DBlock.dimensionSizeArray = dimensionSizeArray

			weight5DTensor = NewAutomaticConvolution3DBlock:generateWeightTensor(dimensionSizeArray)

			NewAutomaticConvolution3DBlock:setWeightTensor(weight5DTensor, true)

		elseif (not dimensionSizeArray) then

			dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weight5DTensor)

		end

		for a = 1, transformedTensorDimensionSizeArray[1], 1 do

			local subInputTensor = inputTensor[a]

			transformedTensor[a] = {}

			for w, weight4DTensor in ipairs(weight5DTensor) do

				transformedTensor[a][w] = {}

				BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function()

					for c = 1, transformedTensorDimensionSizeArray[3], 1 do

						transformedTensor[a][w][c] = {}

						for d = 1, transformedTensorDimensionSizeArray[4], 1 do

							transformedTensor[a][w][c][d] = {}

							for e = 1, transformedTensorDimensionSizeArray[5], 1 do

								local originDimensionIndexArray = {1, (c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

								local targetDimensionIndexArray = {dimensionSizeArray[2], (c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2], (e - 1) * strideDimensionSizeArray[3] + kernelDimensionSizeArray[3]}

								local extractedInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

								local subZTensor = AqwamTensorLibrary:multiply(extractedInputTensor, weight4DTensor)

								transformedTensor[a][w][c][d][e] = AqwamTensorLibrary:sum(subZTensor)

							end

						end

					end

				end)

			end

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return transformedTensor

	end)

	NewAutomaticConvolution3DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

		local kernelDimensionSizeArray = NewAutomaticConvolution3DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewAutomaticConvolution3DBlock.strideDimensionSizeArray

		local weight5DTensor = NewAutomaticConvolution3DBlock:getWeightTensor(true)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		local initialPartialFirstDerivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local numberOfInputChannels = inputTensorDimensionSizeArray[2]

		local coroutineArray = {}

		for kernelIndex, weight4DTensor in ipairs(weight5DTensor) do -- So many for loops.

			BaseConvolutionBlock:createCoroutineToArray(coroutineArray, function() -- The calculation here is so slow that I am forced to use coroutines here. What the fuck.

				for a = 1, initialPartialFirstDerivativeTensorSizeArray[1], 1 do

					for c = 1, initialPartialFirstDerivativeTensorSizeArray[3], 1 do

						for d = 1, initialPartialFirstDerivativeTensorSizeArray[4], 1 do

							for e = 1, initialPartialFirstDerivativeTensorSizeArray[5] do

								local initialPartialFirstDerivativeValue = initialPartialFirstDerivativeTensor[a][kernelIndex][c][d][e]

								local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2], (e - 1) * strideDimensionSizeArray[3]}

								--[[
							
									Since the target dimension index array can be determined by adding the kernel dimension size array to the origin index array, we don't need to find the target dimension index array. 
									Also we don't need to add 1 since the the kernel dimension size array "for" loop iteration will start with 1 and add to the origin dimension index array.
							
								--]]

								local subPartialFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeValue, weight4DTensor)

								for inputChannelIndex = 1, numberOfInputChannels, 1 do

									for i = 1, kernelDimensionSizeArray[1], 1 do

										for j = 1, kernelDimensionSizeArray[2], 1 do

											for k = 1, kernelDimensionSizeArray[3], 1 do

												chainRuleFirstDerivativeTensor[a][inputChannelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j][originDimensionIndexArray[3] + k] = chainRuleFirstDerivativeTensor[a][inputChannelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j][originDimensionIndexArray[3] + k] + subPartialFirstDerivativeTensor[inputChannelIndex][i][j][k]

											end

										end

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

	NewAutomaticConvolution3DBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]

		local initialPartialFirstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)

		local kernelDimensionSizeArray = NewAutomaticConvolution3DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewAutomaticConvolution3DBlock.strideDimensionSizeArray

		local dimensionSizeArray = NewAutomaticConvolution3DBlock.dimensionSizeArray

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
								
								for e = 1, initialPartialFirstDerivativeTensorDimensionSizeArray[5], 1 do
									
								local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

									local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + kernelDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2] + kernelDimensionSizeArray[2], (e - 1) * strideDimensionSizeArray[3] + kernelDimensionSizeArray[3]}

									local extractedSubInputTensor = AqwamTensorLibrary:extract(subInputTensor, originDimensionIndexArray, targetDimensionIndexArray)

									local subNewFirstDerivativeTensor = AqwamTensorLibrary:multiply(extractedSubInputTensor, subInitialPartialFirstDerivativeTensor[c][d][e])

									firstDerivativeTensor[kernelIndex][kernelChannelIndex] = AqwamTensorLibrary:add(firstDerivativeTensor[kernelIndex][kernelChannelIndex], subNewFirstDerivativeTensor)
									
								end

							end

						end

					end

				end)

			end

		end

		BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

		return {firstDerivativeTensor}

	end)

	return NewAutomaticConvolution3DBlock

end

return AutomaticConvolution3DBlock
