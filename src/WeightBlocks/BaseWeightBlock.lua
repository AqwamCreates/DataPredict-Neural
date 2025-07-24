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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseFunctionBlock = require(script.Parent.Parent.Cores.BaseFunctionBlock)

BaseWeightBlock = {}

BaseWeightBlock.__index = BaseWeightBlock

setmetatable(BaseWeightBlock, BaseFunctionBlock)

local defaultLearningRate = 0.01

local defaultWeightInitializationMode = "RandomUniform"

local defaultUpdateWeightTensorInPlace = true

local function performInPlaceSubtraction(tensorToUpdate, tensorToUseForUpdate, dimensionSizeArray, numberOfDimensions, currentDimension) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do performInPlaceSubtraction(tensorToUpdate[i], tensorToUseForUpdate[i], dimensionSizeArray, numberOfDimensions, nextDimension) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do tensorToUpdate[i] = (tensorToUpdate[i] - tensorToUseForUpdate[i]) end

	end

end

local function performInPlaceAddition(tensorToUpdate, tensorToUseForUpdate, dimensionSizeArray, numberOfDimensions, currentDimension) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do performInPlaceAddition(tensorToUpdate[i], tensorToUseForUpdate[i], dimensionSizeArray, numberOfDimensions, nextDimension) end

	else

		for i = 1, dimensionSizeArray[currentDimension], 1 do tensorToUpdate[i] = (tensorToUpdate[i] + tensorToUseForUpdate[i]) end

	end

end

local function performInPlaceUpdate(inPlaceUpdateFunction, weightTensor, weightLossTensor)

	if (type(weightLossTensor) == "number") then error("The weight loss tensor must not be a number in order to use in-place weight update operations.") end

	if (type(weightTensor) == "number") then error("The weight tensor must not be a number in order to use in-place weight update operations.") end

	local weightLossTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightLossTensor)

	local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightTensor)

	local weightLossTensorNumberOfDimensions = #weightLossTensorDimensionSizeArray

	local weightTensorNumberOfDimensions = #weightTensorDimensionSizeArray

	if (weightTensorNumberOfDimensions ~= weightLossTensorNumberOfDimensions) then error("The weight tensor and the weight loss tensor does not have equal number of dimensions. The weight tensor has dimension of " .. weightTensorNumberOfDimensions ..", but weight loss tensor has the dimension of " .. weightLossTensorNumberOfDimensions .. ".") end

	for i, weightTensorDimensionSize in ipairs(weightTensorDimensionSizeArray) do

		local weightLossTensorDimensionSize = weightLossTensorDimensionSizeArray[i]

		if (weightTensorDimensionSize ~= weightLossTensorDimensionSize) then error("The weight tensor and the weight loss tensor does not have equal size at dimension " .. i .. ". The weight tensor has the size of " .. weightTensorDimensionSize ..", but weight loss tensor has the size of " .. weightLossTensorDimensionSize .. ".") end

	end

	inPlaceUpdateFunction(weightTensor, weightLossTensor, weightTensorDimensionSizeArray, weightTensorNumberOfDimensions, 1)

end

function BaseWeightBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseWeightBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBaseWeightBlock, BaseWeightBlock)
	
	NewBaseWeightBlock:setName("BaseWeightBlock")
	
	NewBaseWeightBlock:setClassName("WeightBlock")
	
	NewBaseWeightBlock:setSaveInputTensorArray(true)
	
	NewBaseWeightBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	NewBaseWeightBlock.dimensionSizeArray = parameterDictionary.dimensionSizeArray
	
	NewBaseWeightBlock.weightInitializationMode = parameterDictionary.weightInitializationMode or defaultWeightInitializationMode
	
	NewBaseWeightBlock.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewBaseWeightBlock.Optimizer = parameterDictionary.Optimizer
	
	NewBaseWeightBlock.Regularizer = parameterDictionary.Regularizer
	
	NewBaseWeightBlock.lowestValue = parameterDictionary.lowestValue
	
	NewBaseWeightBlock.highestValue = parameterDictionary.highestValue
	
	NewBaseWeightBlock.mean = parameterDictionary.mean
	
	NewBaseWeightBlock.standardDeviation = parameterDictionary.standardDeviation

	NewBaseWeightBlock.updateWeightTensorInPlace = parameterDictionary.updateWeightTensorInPlace or defaultUpdateWeightTensorInPlace
	
	return NewBaseWeightBlock
	
end

function BaseWeightBlock:setLearningRate(learningRate)

	self.learningRate = learningRate or defaultLearningRate

end

function BaseWeightBlock:getLearningRate()

	return self.learningRate

end

function BaseWeightBlock:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function BaseWeightBlock:getOptimizer()
	
	return self.Optimizer
	
end

function BaseWeightBlock:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function BaseWeightBlock:getRegularizer()

	return self.Regularizer

end

function BaseWeightBlock:setWeightTensor(weightTensor, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.weightTensor = weightTensor
		
	else
		
		self.weightTensor = self:deepCopyTable(weightTensor)
		
	end
	
end

function BaseWeightBlock:getWeightTensor(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.weightTensor
		
	else
		
		return self:deepCopyTable(self.weightTensor)

	end

end

function BaseWeightBlock:generateWeightTensor(dimensionSizeArray, numberOfInputNeurons, numberOfOutputNeurons)
	
	local dimensionSizeArray = dimensionSizeArray or self.dimensionSizeArray
	
	local initializationMode = self.weightInitializationMode

	if (not dimensionSizeArray) then error("No dimension size array for weight initialization!") end
	
	local numberOfDimensions = #dimensionSizeArray
	
	if (not numberOfInputNeurons) then
		
		numberOfInputNeurons = 1

		for i = 1, (numberOfDimensions - 1), 1 do numberOfInputNeurons = numberOfInputNeurons * dimensionSizeArray[i] end
		
	end
	
	if (not numberOfOutputNeurons) then numberOfOutputNeurons = dimensionSizeArray[numberOfDimensions] end

	if (initializationMode == "Zero") then

		return AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

	elseif (initializationMode == "RandomUniform") then

		return AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, self.lowestValue, self.highestValue)

	elseif (initializationMode == "RandomNormal") then

		return AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, self.mean, self.standardDeviation)

	elseif (initializationMode == "HeNormal") then

		local variancePart1 = 2 / numberOfInputNeurons

		local variancePart = math.sqrt(variancePart1)

		local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomNormalTensor)

	elseif (initializationMode == "HeUniform") then

		local variancePart1 = 6 / numberOfInputNeurons

		local variancePart = math.sqrt(variancePart1)

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomUniformTensor) 

	elseif (initializationMode == "XavierNormal") then

		local variancePart1 = 2 / (numberOfInputNeurons + numberOfOutputNeurons)

		local variancePart = math.sqrt(variancePart1)

		local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomNormalTensor) 

	elseif (initializationMode == "XavierUniform") then

		local variancePart1 = 6 / (numberOfInputNeurons + numberOfInputNeurons)

		local variancePart = math.sqrt(variancePart1)

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomUniformTensor)

	elseif (initializationMode == "LeCunNormal") then

		local variancePart1 = 1 / numberOfInputNeurons

		local variancePart = math.sqrt(variancePart1)

		local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomNormalTensor) 

	elseif (initializationMode == "LeCunUniform") then

		local variancePart1 = 3 / numberOfInputNeurons

		local variancePart = math.sqrt(variancePart1)

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomUniformTensor) 
		
	elseif (initializationMode == "None") then
		
		return nil
		
	else
		
		error("Invalid weight initialization mode.")

	end
	
end

function BaseWeightBlock:gradientDescent(weightLossTensor)

	local weightTensor = self.weightTensor

	local learningRate = self.learningRate

	local Optimizer = self.Optimizer

	local Regularizer = self.Regularizer

	if (Regularizer) then

		local regularizationTensor = Regularizer:calculate(weightTensor)

		weightLossTensor = AqwamTensorLibrary:add(weightLossTensor, regularizationTensor)

	end

	if (Optimizer) then

		weightLossTensor = Optimizer:calculate(learningRate, weightLossTensor)

	else

		weightLossTensor = AqwamTensorLibrary:multiply(learningRate, weightLossTensor)

	end
	
	if (self.updateWeightTensorInPlace) then

		performInPlaceUpdate(performInPlaceSubtraction, weightTensor, weightLossTensor)
		
	else
		
		self.weightTensor = AqwamTensorLibrary:subtract(weightTensor, weightLossTensor)
		
	end

end

function BaseWeightBlock:gradientAscent(weightLossTensor)

	local weightTensor = self.weightTensor

	local learningRate = self.learningRate

	local Optimizer = self.Optimizer

	local Regularizer = self.Regularizer

	if (Regularizer) then

		local regularizationTensor = Regularizer:calculate(weightTensor)

		weightLossTensor = AqwamTensorLibrary:add(weightLossTensor, regularizationTensor)

	end

	if (Optimizer) then

		weightLossTensor = Optimizer:calculate(learningRate, weightLossTensor)

	else

		weightLossTensor = AqwamTensorLibrary:multiply(learningRate, weightLossTensor)

	end

	if (self.updateWeightTensorInPlace) then

		performInPlaceUpdate(performInPlaceAddition, weightTensor, weightLossTensor)

	else

		self.weightTensor = AqwamTensorLibrary:add(weightTensor, weightLossTensor)

	end

end

return BaseWeightBlock