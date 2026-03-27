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

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

local SoftActorCriticModel = {}

SoftActorCriticModel.__index = SoftActorCriticModel

setmetatable(SoftActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local defaultAlpha = 0.1

local defaultAveragingRate = 0.995

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

local function calculateCategoricalProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

	return targetActionTensor

end

local function calculateActionTensor(meanTensor, standardDeviationTensor, noiseTensor)
	
	local actionVectoPart1 = AqwamTensorLibrary:multiply(standardDeviationTensor, noiseTensor)
	
	local actionTensor = AqwamTensorLibrary:add(meanTensor, actionVectoPart1)
	
	return actionTensor
	
end

local function calculateDiagonalGaussianProbability(meanTensor, standardDeviationTensor, noiseTensor)

	local valueTensorPart1 = AqwamTensorLibrary:multiply(standardDeviationTensor, noiseTensor)

	local valueTensor = AqwamTensorLibrary:add(meanTensor, valueTensorPart1)

	local zScoreTensorPart1 = AqwamTensorLibrary:subtract(valueTensor, meanTensor)

	local zScoreTensor = AqwamTensorLibrary:divide(zScoreTensorPart1, standardDeviationTensor)

	local squaredZScoreTensor = AqwamTensorLibrary:power(zScoreTensor, 2)

	local logValueTensorPart1 = AqwamTensorLibrary:logarithm(standardDeviationTensor)

	local logValueTensorPart2 = AqwamTensorLibrary:multiply(2, logValueTensorPart1)

	local logValueTensorPart3 = AqwamTensorLibrary:add(squaredZScoreTensor, logValueTensorPart2)

	local logValueTensor = AqwamTensorLibrary:add(logValueTensorPart3, math.log(2 * math.pi))

	logValueTensor = AqwamTensorLibrary:multiply(-0.5, logValueTensor)

	return logValueTensor

end

local function calculateDiagonalGaussianProbabilityGradient(meanTensor, standardDeviationTensor, noiseTensor)

	local actionTensorPart1 = AqwamTensorLibrary:multiply(standardDeviationTensor, noiseTensor)

	local actionTensor = AqwamTensorLibrary:add(meanTensor, actionTensorPart1)

	local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, meanTensor)

	local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(standardDeviationTensor, 2)

	local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

	return actionProbabilityGradientTensor

end

function SoftActorCriticModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSoftActorCritic = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewSoftActorCritic, SoftActorCriticModel)
	
	NewSoftActorCritic:setName("SoftActorCritic")
	
	NewSoftActorCritic.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewSoftActorCritic.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewSoftActorCritic.PrimaryCriticWeightTensorArrayArray = parameterDictionary.PrimaryCriticWeightTensorArrayArray or {}
	
	NewSoftActorCritic.TargetCriticWeightTensorArrayArray = parameterDictionary.TargetCriticWeightTensorArrayArray or {}
	
	NewSoftActorCritic:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local ActorModel = NewSoftActorCritic.ActorModel

		local previousActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, true)
		
		local currentActionTensor = ActorModel:forwardPropagate(currentFeatureTensor, true)

		local previousActionProbabilityTensor = calculateCategoricalProbability(previousActionTensor)
		
		local currentActionProbabilityTensor = calculateCategoricalProbability(currentActionTensor)
		
		local ClassesList = ActorModel:getClassesList()
		
		local previousClassIndex = table.find(ClassesList, previousAction)

		local previousActionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			previousActionProbabilityGradientTensor[i] = (((i == previousClassIndex) and 1) or 0) - previousActionProbabilityTensor[1][i]

		end
		
		previousActionProbabilityGradientTensor = {previousActionProbabilityGradientTensor}

		local previousLogActionProbabilityTensor = AqwamTensorLibrary:logarithm(previousActionProbabilityTensor)
		
		local currentLogActionProbabilityTensor = AqwamTensorLibrary:logarithm(currentActionProbabilityTensor)
		
		return NewSoftActorCritic:update(previousFeatureTensor, previousActionTensor, previousLogActionProbabilityTensor, previousActionProbabilityGradientTensor, rewardValue, currentFeatureTensor, currentActionTensor, currentLogActionProbabilityTensor, currentAction, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)
		
		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionNoiseTensor)
		
		local currentActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)
		
		local previousActionTensor = calculateActionTensor(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)
		
		local currentActionTensor = calculateActionTensor(currentActionMeanTensor, previousActionStandardDeviationTensor, currentActionNoiseTensor)
		
		local previousLogActionProbabilityTensor = calculateDiagonalGaussianProbability(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)
		
		local currentLogActionProbabilityTensor = calculateDiagonalGaussianProbability(currentActionMeanTensor, previousActionStandardDeviationTensor, currentActionNoiseTensor)
		
		local previousActionProbabilityGradientTensor = calculateDiagonalGaussianProbabilityGradient(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)
		
		return NewSoftActorCritic:update(previousFeatureTensor, previousActionTensor, previousLogActionProbabilityTensor, previousActionProbabilityGradientTensor, rewardValue, currentFeatureTensor, currentActionTensor, currentLogActionProbabilityTensor, nil, terminalStateValue)
		
	end)
	
	NewSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewSoftActorCritic:setResetFunction(function() end)
	
	return NewSoftActorCritic
	
end

function SoftActorCriticModel:update(previousFeatureTensor, previousActionTensor, previousLogActionProbabilityTensor, previousActionProbabilityGradientTensor, rewardValue, currentFeatureTensor, currentActionTensor, currentLogActionProbabilityTensor, currentAction, terminalStateValue)
	
	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha
	
	local averagingRate = self.averagingRate
	
	local PrimaryCriticWeightTensorArrayArray = self.PrimaryCriticWeightTensorArrayArray

	local TargetCriticWeightTensorArrayArray = self.TargetCriticWeightTensorArrayArray
	
	local currentLogActionProbabilityValue
	
	if (currentAction) then
		
		local ClassesList = ActorModel:getClassesList()
		
		local actionIndex = table.find(ClassesList, currentAction)
		
		currentLogActionProbabilityValue = currentLogActionProbabilityTensor[1][actionIndex]
		
	else
		
		currentLogActionProbabilityValue = AqwamTensorLibrary:sum(currentLogActionProbabilityTensor)
		
	end

	local targetCurrentCriticValueArray = {}
	
	local concatenatedCurrentFeatureAndActionTensor = AqwamTensorLibrary:concatenate(currentFeatureTensor, currentActionTensor, 2)

	for i = 1, 2, 1 do 

		CriticModel:setWeightTensorArray(TargetCriticWeightTensorArrayArray[i])

		targetCurrentCriticValueArray[i] = CriticModel:forwardPropagate(concatenatedCurrentFeatureAndActionTensor)[1][1]
		
		TargetCriticWeightTensorArrayArray[i] = TargetCriticWeightTensorArrayArray[i] or CriticModel:getWeightTensorArray(true)

	end

	local minimumTargetCurrentCriticValue = math.min(table.unpack(targetCurrentCriticValueArray))
	
	local yValuePart1 = (1 - terminalStateValue) * (minimumTargetCurrentCriticValue - (alpha * currentLogActionProbabilityValue))

	local yValue = rewardValue + (self.discountFactor * yValuePart1)

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2}, 0)
	
	local concatenatedPreviousFeatureAndActionTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, previousActionTensor, 2)
	
	local previousCriticValueArray = {}

	for i = 1, 2, 1 do

		CriticModel:setWeightTensorArray(PrimaryCriticWeightTensorArrayArray[i], true)

		local primaryPreviousCriticValue = CriticModel:forwardPropagate(concatenatedPreviousFeatureAndActionTensor, true)[1][1] 

		local criticLoss = 2 * (primaryPreviousCriticValue - yValue)
		
		previousCriticValueArray[i] = primaryPreviousCriticValue

		temporalDifferenceErrorTensor[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

		CriticModel:update(criticLoss, true)
		
		local TargetWeightTensorArray = TargetCriticWeightTensorArrayArray[i]
		
		local PrimaryWeightTensorArray = CriticModel:getWeightTensorArray(true)
		
		PrimaryCriticWeightTensorArrayArray[i] = PrimaryWeightTensorArray
		
		TargetCriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	end
	
	local minimumPreviousCriticValue = math.min(table.unpack(previousCriticValueArray))

	local actorLossTensorPart1 = AqwamTensorLibrary:multiply(alpha, previousLogActionProbabilityTensor)

	local actorLossTensorPart2 = AqwamTensorLibrary:subtract(minimumPreviousCriticValue, actorLossTensorPart1)
	
	local actorLossTensor = AqwamTensorLibrary:multiply(actorLossTensorPart2, previousActionProbabilityGradientTensor)
	
	actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

	ActorModel:forwardPropagate(previousFeatureTensor, true)

	ActorModel:update(actorLossTensor, true)
	
	return temporalDifferenceErrorTensor
	
end

function SoftActorCriticModel:setPrimaryCriticWeightTensorArray1(PrimaryCriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticWeightTensorArrayArray[1] = PrimaryCriticWeightTensorArray1

	else

		self.PrimaryCriticWeightTensorArrayArray[1] = self:deepCopyTable(PrimaryCriticWeightTensorArray1)

	end

end

function SoftActorCriticModel:setPrimaryCriticWeightTensorArray2(PrimaryCriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticWeightTensorArrayArray[2] = PrimaryCriticWeightTensorArray2

	else

		self.PrimaryCriticWeightTensorArrayArray[2] = self:deepCopyTable(PrimaryCriticWeightTensorArray2)

	end

end

function SoftActorCriticModel:setTargetCriticWeightTensorArray1(TargetCriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticWeightTensorArrayArray[1] = TargetCriticWeightTensorArray1

	else

		self.TargetCriticWeightTensorArrayArray[1] = self:deepCopyTable(TargetCriticWeightTensorArray1)

	end

end

function SoftActorCriticModel:setTargetCriticWeightTensorArray2(TargetCriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticWeightTensorArrayArray[2] = TargetCriticWeightTensorArray2

	else

		self.TargetCriticWeightTensorArrayArray[2] = self:deepCopyTable(TargetCriticWeightTensorArray2)

	end

end

function SoftActorCriticModel:getPrimaryCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.PrimaryCriticWeightTensorArrayArray[1])

	end

end

function SoftActorCriticModel:getPrimaryCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.PrimaryCriticWeightTensorArrayArray[2])

	end

end

function SoftActorCriticModel:getTargetCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.TargetCriticWeightTensorArrayArray[1])

	end

end

function SoftActorCriticModel:getTargetCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.TargetCriticWeightTensorArrayArray[2])

	end

end

return SoftActorCriticModel
