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

local DualRecurrentReinforcementLearningActorCriticBaseModel = require(script.Parent.DualRecurrentReinforcementLearningActorCriticBaseModel)

RecurrentSoftActorCriticModel = {}

RecurrentSoftActorCriticModel.__index = RecurrentSoftActorCriticModel

setmetatable(RecurrentSoftActorCriticModel, DualRecurrentReinforcementLearningActorCriticBaseModel)

local defaultAlpha = 0.1

local defaultAveragingRate = 0.995

local function calculateProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

	return targetActionTensor

end

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

local function calculateLogActionProbabilityTensor(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

	local actionTensorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

	local actionTensor = AqwamTensorLibrary:add(actionMeanTensor, actionTensorPart1)

	local zScoreTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, actionMeanTensor)

	local zScoreTensor = AqwamTensorLibrary:divide(zScoreTensorPart1, actionStandardDeviationTensor)

	local squaredZScoreTensor = AqwamTensorLibrary:power(zScoreTensor, 2)

	local logActionProbabilityTensorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationTensor)

	local logActionProbabilityTensorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityTensorPart1)

	local logActionProbabilityTensorPart3 = AqwamTensorLibrary:add(squaredZScoreTensor, logActionProbabilityTensorPart2)

	local logActionProbabilityTensorPart4 = AqwamTensorLibrary:add(logActionProbabilityTensorPart3, math.log(2 * math.pi))

	local logActionProbabilityTensor = AqwamTensorLibrary:multiply(-0.5, logActionProbabilityTensorPart4)

	return logActionProbabilityTensor

end

function RecurrentSoftActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentSoftActorCritic = DualRecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentSoftActorCritic, RecurrentSoftActorCriticModel)

	NewRecurrentSoftActorCritic:setName("RecurrentSoftActorCritic")

	NewRecurrentSoftActorCritic.alpha = parameterDictionary.alpha or defaultAlpha

	NewRecurrentSoftActorCritic.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentSoftActorCritic.CriticWeightTensorArrayArray = parameterDictionary.CriticWeightTensorArrayArray or {}

	NewRecurrentSoftActorCritic:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local ActorModel = NewRecurrentSoftActorCritic.ActorModel
		
		local CriticModel = NewRecurrentSoftActorCritic.CriticModel
		
		local ClassesList = ActorModel:getClassesList()
		
		local outputDimensionSizeArray = {1, #ClassesList}
		
		local actorHiddenStateTensorArray = NewRecurrentSoftActorCritic.actorHiddenStateTensorArray

		local actorHiddenStateTensor = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local previousActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)

		local currentActionTensor = ActorModel:forwardPropagate(currentFeatureTensor, previousActionTensor)

		local previousActionProbabilityTensor = calculateProbability(previousActionTensor)

		local currentActionProbabilityTensor = calculateProbability(currentActionTensor)

		local previousLogActionProbabilityTensor = AqwamTensorLibrary:logarithm(previousActionProbabilityTensor)

		local currentLogActionProbabilityTensor = AqwamTensorLibrary:logarithm(currentActionProbabilityTensor)
		
		actorHiddenStateTensorArray[1] = previousActionTensor

		return NewRecurrentSoftActorCritic:backwardPropagate(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

	end)

	NewRecurrentSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensordimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensordimensionSizeArray)

		end

		local currentActionMeanTensor = NewRecurrentSoftActorCritic.ActorModel:forwardPropagate(currentFeatureTensor, actionMeanTensor)

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionNoiseTensor)

		local currentActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		local previousLogActionProbabilityTensor = calculateLogActionProbabilityTensor(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

		local currentLogActionProbabilityTensor = calculateLogActionProbabilityTensor(currentActionMeanTensor, actionStandardDeviationTensor, currentActionNoiseTensor)
		
		NewRecurrentSoftActorCritic.actorHiddenStateTensorArray[1] = actionMeanTensor

		return NewRecurrentSoftActorCritic:backwardPropagate(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, nil, rewardValue, currentFeatureTensor, terminalStateValue)

	end)

	NewRecurrentSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)

	NewRecurrentSoftActorCritic:setResetFunction(function() end)

	return NewRecurrentSoftActorCritic

end

function RecurrentSoftActorCriticModel:backwardPropagate(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
	
	local actorHiddenStateTensorArray = self.actorHiddenStateTensorArray

	local criticHiddenStateValueArray = self.criticHiddenStateValueArray
	
	local CriticWeightTensorArrayArray = self.CriticWeightTensorArrayArray

	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha

	local averagingRate = self.averagingRate

	local PreviousCriticWeightTensorArrayArray = {}

	local previousLogActionProbabilityValue
	
	criticHiddenStateValueArray[1] = criticHiddenStateValueArray[1] or 0

	criticHiddenStateValueArray[2] = criticHiddenStateValueArray[2] or 0

	if (action) then

		local ClassesList = ActorModel:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		previousLogActionProbabilityValue = previousLogActionProbabilityTensor[1][actionIndex]

	else

		previousLogActionProbabilityValue = AqwamTensorLibrary:sum(previousLogActionProbabilityTensor)

	end

	local currentCriticValueArray = {}

	for i = 1, 2, 1 do 

		CriticModel:setWeightTensorArray(CriticWeightTensorArrayArray[i])
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, {{criticHiddenStateValueArray[i]}})

		currentCriticValueArray[i] = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1] 

		local CriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

		PreviousCriticWeightTensorArrayArray[i] = CriticWeightTensorArray

	end

	local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))

	local yValuePart1 = (1 - terminalStateValue) * (minimumCurrentCriticValue - (alpha * previousLogActionProbabilityValue))

	local yValue = rewardValue + (self.discountFactor * yValuePart1)

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2}, 0)

	local previousCriticValueArray = {}

	for i = 1, 2, 1 do

		CriticModel:setWeightTensorArray(PreviousCriticWeightTensorArrayArray[i], true)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, {{criticHiddenStateValueArray[i]}})[1][1] 

		local criticLoss = previousCriticValue - yValue

		temporalDifferenceErrorTensor[1][i] = -criticLoss  -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

		previousCriticValueArray[i] = previousCriticValue

		CriticModel:update(criticLoss)

		local TargetWeightTensorArray = CriticModel:getWeightTensorArray(true)

		CriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PreviousCriticWeightTensorArrayArray[i])
		
		criticHiddenStateValueArray[i] = previousCriticValue

	end

	local minimumCurrentCriticValue = math.min(table.unpack(previousCriticValueArray))

	local actorLossTensor = AqwamTensorLibrary:multiply(alpha, previousLogActionProbabilityTensor)

	actorLossTensor = AqwamTensorLibrary:subtract(minimumCurrentCriticValue, actorLossTensor)

	actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

	ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensorArray[1])

	ActorModel:update(actorLossTensor)

	return temporalDifferenceErrorTensor

end

function RecurrentSoftActorCriticModel:setCrtiticWeightTensorArray1(CriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticWeightTensorArrayArray[1] = CriticWeightTensorArray1

	else

		self.CriticWeightTensorArrayArray[1] = self:deepCopyTable(CriticWeightTensorArray1)

	end

end

function RecurrentSoftActorCriticModel:setCriticWeightTensorArray2(CriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticWeightTensorArrayArray[2] = CriticWeightTensorArray2

	else

		self.CriticWeightTensorArrayArray[2] = self:deepCopyTable(CriticWeightTensorArray2)

	end

end

function RecurrentSoftActorCriticModel:getCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.CriticWeightTensorArrayArray[1])

	end

end

function RecurrentSoftActorCriticModel:getCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.CriticWeightTensorArrayArray[2])

	end

end

return RecurrentSoftActorCriticModel