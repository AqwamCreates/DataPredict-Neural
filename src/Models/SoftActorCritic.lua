--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
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

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

local function calculateCategoricalProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

	return targetActionTensor

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

function SoftActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewSoftActorCritic = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewSoftActorCritic, SoftActorCriticModel)

	NewSoftActorCritic:setName("SoftActorCritic")

	NewSoftActorCritic.alpha = parameterDictionary.alpha or defaultAlpha

	NewSoftActorCritic.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewSoftActorCritic.CriticModelParametersArray = parameterDictionary.CriticModelParametersArray or {}

	NewSoftActorCritic:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewSoftActorCritic.ActorModel

		local CriticModel = NewSoftActorCritic.CriticModel

		local previousActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, true)

		local currentActionTensor = ActorModel:forwardPropagate(currentFeatureTensor, true)

		local previousActionProbabilityTensor = calculateCategoricalProbability(previousActionTensor)

		local currentActionProbabilityTensor = calculateCategoricalProbability(currentActionTensor)

		local previousLogActionProbabilityTensor = AqwamTensorLibrary:logarithm(previousActionProbabilityTensor)

		local currentLogActionProbabilityTensor = AqwamTensorLibrary:logarithm(currentActionProbabilityTensor)

		return NewSoftActorCritic:update(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue)

	end)

	NewSoftActorCritic:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionNoiseTensor)

		local currentActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		local previousLogActionProbabilityTensor = calculateDiagonalGaussianProbability(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionStandardDeviationTensor)

		local currentLogActionProbabilityTensor = calculateDiagonalGaussianProbability(currentActionMeanTensor, previousActionStandardDeviationTensor, currentActionNoiseTensor)

		return NewSoftActorCritic:update(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, nil, rewardValue, currentFeatureTensor, terminalStateValue)

	end)

	NewSoftActorCritic:setEpisodeUpdateFunction(function(terminalStateValue) end)

	NewSoftActorCritic:setResetFunction(function() end)

	return NewSoftActorCritic

end

function SoftActorCriticModel:update(previousFeatureTensor, previousLogActionProbabilityTensor, currentLogActionProbabilityTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue)

	local CriticModelParametersArray = self.CriticModelParametersArray

	local CriticModel = self.CriticModel

	local ActorModel = self.ActorModel

	local alpha = self.alpha

	local averagingRate = self.averagingRate

	local PreviousCriticModelParametersArray = {}

	local previousLogActionProbabilityValue

	if (previousAction) then

		local ClassesList = ActorModel:getClassesList()

		local actionIndex = table.find(ClassesList, previousAction)

		previousLogActionProbabilityValue = previousLogActionProbabilityTensor[1][actionIndex]

	else

		previousLogActionProbabilityValue = AqwamTensorLibrary:sum(previousLogActionProbabilityTensor)

	end

	local currentCriticValueArray = {}

	for i = 1, 2, 1 do 

		CriticModel:setWeightTensorArray(CriticModelParametersArray[i])

		currentCriticValueArray[i] = CriticModel:forwardPropagate(currentFeatureTensor)[1][1] 

		local CriticModelParameters = CriticModel:getWeightTensorArray(true)

		PreviousCriticModelParametersArray[i] = CriticModelParameters

	end

	local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))

	local yValuePart1 = (1 - terminalStateValue) * (minimumCurrentCriticValue - (alpha * previousLogActionProbabilityValue))

	local yValue = rewardValue + (self.discountFactor * yValuePart1)

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2}, 0)

	local previousCriticValueArray = {}

	for i = 1, 2, 1 do 

		CriticModel:setWeightTensorArray(PreviousCriticModelParametersArray[i], true)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, true)[1][1] 

		local criticLoss = previousCriticValue - yValue

		temporalDifferenceErrorTensor[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

		previousCriticValueArray[i] = previousCriticValue

		CriticModel:update(criticLoss, true)

		local TargetModelParameters = CriticModel:getWeightTensorArray(true)

		CriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetModelParameters, PreviousCriticModelParametersArray[i])

	end

	local minimumCurrentCriticValue = math.min(table.unpack(previousCriticValueArray))

	local actorLossTensor = AqwamTensorLibrary:multiply(alpha, previousLogActionProbabilityTensor)

	actorLossTensor = AqwamTensorLibrary:subtract(minimumCurrentCriticValue, actorLossTensor)

	actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

	ActorModel:forwardPropagate(previousFeatureTensor, true)

	ActorModel:update(actorLossTensor, true)

	return temporalDifferenceErrorTensor

end

function SoftActorCriticModel:setCrtiticModelParameters1(CriticModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticModelParametersArray[1] = CriticModelParameters1

	else

		self.CriticModelParametersArray[1] = self:deepCopyTable(CriticModelParameters1)

	end

end

function SoftActorCriticModel:setCriticModelParameters2(CriticModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticModelParametersArray[2] = CriticModelParameters2

	else

		self.CriticModelParametersArray[2] = self:deepCopyTable(CriticModelParameters2)

	end

end

function SoftActorCriticModel:getCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticModelParametersArray[1]

	else

		return self:deepCopyTable(self.CriticModelParametersArray[1])

	end

end

function SoftActorCriticModel:getCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticModelParametersArray[2]

	else

		return self:deepCopyTable(self.CriticModelParametersArray[2])

	end

end

return SoftActorCriticModel
