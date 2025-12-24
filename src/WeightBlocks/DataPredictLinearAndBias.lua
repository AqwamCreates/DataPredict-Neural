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

local BaseWeightBlock = require(script.Parent.BaseWeightBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local DataPredictLinearAndBiasBlock = {}

DataPredictLinearAndBiasBlock.__index = DataPredictLinearAndBiasBlock

setmetatable(DataPredictLinearAndBiasBlock, BaseWeightBlock)

local defaultHasBiasOnCurrentLayer = true

local defaultHasBiasOnNextLayer = true

local function getValueOrDefaultValue(value, defaultValue)

	if (type(value) == "nil") then return defaultValue end

	return value

end

function DataPredictLinearAndBiasBlock.new(parameterDictionary) -- For cross compatibility with DataPredict Library Neural Networks.

	parameterDictionary = parameterDictionary or {}
	
	local hasBiasOnCurrentLayer = getValueOrDefaultValue(parameterDictionary.hasBiasOnCurrentLayer, defaultHasBiasOnCurrentLayer)
	
	local hasBiasOnNextLayer = getValueOrDefaultValue(parameterDictionary.hasBiasOnNextLayer, defaultHasBiasOnNextLayer)
	
	local dimensionSizeArray = parameterDictionary.dimensionSizeArray
	
	dimensionSizeArray[1] = dimensionSizeArray[1] + ((hasBiasOnCurrentLayer and 1) or 0)
	
	dimensionSizeArray[2] = dimensionSizeArray[2] + ((hasBiasOnNextLayer and 1) or 0)
	
	local NewDataPredictLinearAndBiasBlock = BaseWeightBlock.new(parameterDictionary)

	setmetatable(NewDataPredictLinearAndBiasBlock, DataPredictLinearAndBiasBlock)

	NewDataPredictLinearAndBiasBlock:setName("DataPredictLinearAndBiasBlock")

	NewDataPredictLinearAndBiasBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewDataPredictLinearAndBiasBlock.hasBiasOnCurrentLayer = hasBiasOnNextLayer
	
	NewDataPredictLinearAndBiasBlock.hasBiasOnNextLayer = hasBiasOnNextLayer

	NewDataPredictLinearAndBiasBlock:setFunction(function(inputTensorArray)
		
		local inputTensor = inputTensorArray[1]

		local weightTensor = NewDataPredictLinearAndBiasBlock:getWeightTensor(true)

		if (not weightTensor) then

			weightTensor = NewDataPredictLinearAndBiasBlock:generateWeightTensor()

			NewDataPredictLinearAndBiasBlock:setWeightTensor(weightTensor, true)

		end
		
		if (NewDataPredictLinearAndBiasBlock.hasBiasOnCurrentLayer) then

			for data = 1, #inputTensor, 1 do inputTensor[data][1] = 1 end -- Because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron.

		end

		return AqwamTensorLibrary:dotProduct(inputTensorArray[1], weightTensor)

	end)

	NewDataPredictLinearAndBiasBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local weightTensor = NewDataPredictLinearAndBiasBlock:getWeightTensor(true)

		if (not weightTensor) then

			weightTensor = NewDataPredictLinearAndBiasBlock:generateWeightTensor()

			NewDataPredictLinearAndBiasBlock:setWeightTensor(weightTensor, true)

		end

		local weightTensorNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(weightTensor)
		
		local partialFirstDerivativeTensor = AqwamTensorLibrary:copy(initialPartialFirstDerivativeTensor)

		if (NewDataPredictLinearAndBiasBlock.hasBiasOnNextLayer) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

			for i = 1, #partialFirstDerivativeTensor, 1 do partialFirstDerivativeTensor[i][1] = 0 end

		end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(partialFirstDerivativeTensor, AqwamTensorLibrary:transpose(weightTensor, {weightTensorNumberOfDimensions - 1, weightTensorNumberOfDimensions}))

		return {chainRuleFirstDerivativeTensor}

	end)

	NewDataPredictLinearAndBiasBlock:setFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)  

		--[[
		
		We only have calculated the partial derivative and not the full derivative. So another first derivative function is needed to in order to calculate full derivative.
		
		Refer to https://www.3blue1brown.com/lessons/backpropagation-calculus
		
		--]]

		local inputTensor = inputTensorArray[1]
		
		local inputTensorNumberOfDimensions =  AqwamTensorLibrary:getNumberOfDimensions(inputTensor)
		
		local transposedInputTensor = AqwamTensorLibrary:transpose(inputTensor, {inputTensorNumberOfDimensions - 1, inputTensorNumberOfDimensions})
		
		local partialFirstDerivativeTensor = AqwamTensorLibrary:copy(initialPartialFirstDerivativeTensor)
		
		if (NewDataPredictLinearAndBiasBlock.hasBiasOnNextLayer) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

			for i = 1, #partialFirstDerivativeTensor, 1 do partialFirstDerivativeTensor[i][1] = 0 end

		end
		
		local firstDerivativeTensor = AqwamTensorLibrary:dotProduct(transposedInputTensor, partialFirstDerivativeTensor)

		return {firstDerivativeTensor}

	end)

	return NewDataPredictLinearAndBiasBlock

end

function DataPredictLinearAndBiasBlock:gradientDescent(weightLossTensor, numberOfData)

	local weightTensor = self.weightTensor

	local learningRate = self.learningRate

	local Optimizer = self.Optimizer

	local Regularizer = self.Regularizer

	if (Regularizer) then

		local regularizationTensor = Regularizer:calculate(weightTensor)

		weightLossTensor = AqwamTensorLibrary:add(weightLossTensor, regularizationTensor)

	end

	if (numberOfData ~= nil) and (numberOfData ~= 1) then 

		weightLossTensor = AqwamTensorLibrary:divide(weightLossTensor, numberOfData) 

	end

	if (Optimizer) then

		weightLossTensor = Optimizer:calculate(learningRate, weightLossTensor)

	else

		weightLossTensor = AqwamTensorLibrary:multiply(learningRate, weightLossTensor)

	end
	
	local newWeightTensor = AqwamTensorLibrary:subtract(weightTensor, weightLossTensor)
	
	if (self.hasBiasOnNextLayer) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

		for i = 1, #newWeightTensor, 1 do newWeightTensor[i][1] = 0 end

	end

	self.weightTensor = newWeightTensor

end

return DataPredictLinearAndBiasBlock
