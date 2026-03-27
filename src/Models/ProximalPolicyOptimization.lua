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

local ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, ReinforcementLearningActorCriticBaseModel)

local defaultLambda = 0

local defaultUseLogProbabilities = true

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

local function calculateDiagonalGaussianProbabilityGradient(meanTensor, standardDeviationTensor, noiseTensor)

	local actionTensorPart1 = AqwamTensorLibrary:multiply(standardDeviationTensor, noiseTensor)

	local actionTensor = AqwamTensorLibrary:add(meanTensor, actionTensorPart1)

	local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, meanTensor)

	local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(standardDeviationTensor, 2)

	local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

	return actionProbabilityGradientTensor

end

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function ProximalPolicyOptimizationModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewProximalPolicyOptimizationModel = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)

	NewProximalPolicyOptimizationModel:setName("ProximalPolicyOptimization")

	NewProximalPolicyOptimizationModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewProximalPolicyOptimizationModel.useLogProbabilities = NewProximalPolicyOptimizationModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewProximalPolicyOptimizationModel.OldActorModelParameters = parameterDictionary.OldActorModelParameters

	local featureTensorHistory = {}

	local actorGradientTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewProximalPolicyOptimizationModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
		local CurrentActorModelParameters = ActorModel:getModelParameters(true)
		
		local OldModelParameters = NewProximalPolicyOptimizationModel.OldActorModelParameters or CurrentActorModelParameters

		ActorModel:setModelParameters(OldModelParameters, true)

		local oldPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor)
		
		ActorModel:setModelParameters(CurrentActorModelParameters, true)
		
		local currentPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor)
		
		local oldPolicyActionProbabilityTensor = calculateCategoricalProbability(oldPolicyActionTensor)
		
		local currentPolicyActionProbabilityTensor = calculateCategoricalProbability(currentPolicyActionTensor)
		
		local ClassesList = ActorModel:getClassesList()

		local previousActionIndex = table.find(ClassesList, previousAction)
		
		local unwrappedCurrentPolicyActionProbabilityTensor = currentPolicyActionProbabilityTensor[1]
		
		local currentPolicyActionProbability = unwrappedCurrentPolicyActionProbabilityTensor[previousActionIndex]
		
		local oldPolicyActionProbability = oldPolicyActionProbabilityTensor[1][previousActionIndex]
		
		local ratioActionProbability
		
		if (NewProximalPolicyOptimizationModel.useLogProbabilities) then
			
			ratioActionProbability = math.exp(math.log(currentPolicyActionProbability) - math.log(oldPolicyActionProbability))
			
		else
			
			ratioActionProbability = currentPolicyActionProbability / oldPolicyActionProbability
			
		end
		
		local previousActionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			previousActionProbabilityGradientTensor[i] = (((i == previousActionIndex) and 1) or 0) - unwrappedCurrentPolicyActionProbabilityTensor[i]

		end

		previousActionProbabilityGradientTensor = {previousActionProbabilityGradientTensor}

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actorGradientTensorHistory, previousActionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local CurrentActorModelParameters = ActorModel:getModelParameters(true)
		
		local OldModelParameters = NewProximalPolicyOptimizationModel.OldActorModelParameters or CurrentActorModelParameters

		ActorModel:setModelParameters(OldModelParameters, true)

		local oldPolicyActionMeanTensor = ActorModel:forwardPropagate(previousFeatureTensor)
		
		ActorModel:setModelParameters(CurrentActorModelParameters, true)

		local oldPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(oldPolicyActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local currentPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)
		
		local ratioActionProbabiltyTensor
		
		if (NewProximalPolicyOptimizationModel.useLogProbabilities) then

			ratioActionProbabiltyTensor = AqwamTensorLibrary:applyFunction(math.exp, AqwamTensorLibrary:subtract(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor))

		else
			
			currentPolicyActionProbabilityTensor = AqwamTensorLibrary:applyFunction(math.exp, currentPolicyActionProbabilityTensor)
			
			oldPolicyActionProbabilityTensor = AqwamTensorLibrary:applyFunction(math.exp, oldPolicyActionProbabilityTensor)

			ratioActionProbabiltyTensor = AqwamTensorLibrary:divide(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor)

		end
		
		local previousActionProbabilityGradientTensor = calculateDiagonalGaussianProbabilityGradient(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)
		
		previousActionProbabilityGradientTensor = AqwamTensorLibrary:multiply(previousActionProbabilityGradientTensor, ratioActionProbabiltyTensor)
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actorGradientTensorHistory, previousActionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local discountFactor = NewProximalPolicyOptimizationModel.discountFactor

		local lambda = NewProximalPolicyOptimizationModel.lambda
		
		local CurrentActorModelParameters = ActorModel:getModelParameters(true)

		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue)

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, discountFactor)

		for h, featureTensor in ipairs(featureTensorHistory) do

			local actorGradientTensor = actorGradientTensorHistory[h]

			local advantageValue = advantageValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actorGradientTensor, advantageValue)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

			ActorModel:forwardPropagate(featureTensor, true)

			CriticModel:forwardPropagate(featureTensor, true)

			ActorModel:update(actorLossTensor, true)

			CriticModel:update(criticLoss, true)

		end

		table.clear(featureTensorHistory)

		table.clear(actorGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)
		
		NewProximalPolicyOptimizationModel.OldActorModelParameters = CurrentActorModelParameters

	end)

	NewProximalPolicyOptimizationModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actorGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewProximalPolicyOptimizationModel

end

return ProximalPolicyOptimizationModel
