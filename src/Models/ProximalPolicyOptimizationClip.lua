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

local DeepReinforcementLearningActorCriticBaseModel = require(script.Parent.DeepReinforcementLearningActorCriticBaseModel)

local ProximalPolicyOptimizationClipModel = {}

ProximalPolicyOptimizationClipModel.__index = ProximalPolicyOptimizationClipModel

setmetatable(ProximalPolicyOptimizationClipModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local defaultLambda = 0

local defaultUseLogProbabilities = true

local function calculateCategoricalProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentValueTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentValueSumTensor = AqwamTensorLibrary:sum(exponentValueTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentValueTensor, exponentValueSumTensor)

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

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function ProximalPolicyOptimizationClipModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewProximalPolicyOptimizationClipModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewProximalPolicyOptimizationClipModel, ProximalPolicyOptimizationClipModel)

	NewProximalPolicyOptimizationClipModel:setName("ProximalPolicyOptimizationClip")

	NewProximalPolicyOptimizationClipModel.clipRatio = parameterDictionary.clipRatio or defaultClipRatio

	NewProximalPolicyOptimizationClipModel.lambda = parameterDictionary.lambda or defaultLambda

	NewProximalPolicyOptimizationClipModel.useLogProbabilities = NewProximalPolicyOptimizationClipModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters = parameterDictionary.CurrentActorModelParameters

	NewProximalPolicyOptimizationClipModel.OldActorModelParameters = parameterDictionary.OldActorModelParameters

	local featureTensorHistory = {}

	local ratioActionProbabiltyTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewProximalPolicyOptimizationClipModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters = ActorModel:getModelParameters(true)

		ActorModel:setModelParameters(NewProximalPolicyOptimizationClipModel.OldActorModelParameters, true)

		local oldPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor)

		NewProximalPolicyOptimizationClipModel.OldActorModelParameters = ActorModel:getModelParameters(true)

		local oldPolicyActionProbabilityTensor = calculateCategoricalProbability(oldPolicyActionTensor)

		local currentPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor)

		local currentPolicyActionProbabilityTensor = calculateCategoricalProbability(currentPolicyActionTensor)

		ActorModel:setModelParameters(NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters, true)

		local ClassesList = ActorModel:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)

		local ratioActionProbabiltyTensor = table.create(#ClassesList, 0)

		local ratioActionProbability

		if (NewProximalPolicyOptimizationClipModel.useLogProbabilities) then

			ratioActionProbability = math.exp(math.log(currentPolicyActionProbabilityTensor[1][classIndex]) - math.log(oldPolicyActionProbabilityTensor[1][classIndex]))

		else

			ratioActionProbability = currentPolicyActionProbabilityTensor[1][classIndex] / oldPolicyActionProbabilityTensor[1][classIndex]

		end

		ratioActionProbabiltyTensor[classIndex] = ratioActionProbability

		ratioActionProbabiltyTensor = {ratioActionProbabiltyTensor}

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(ratioActionProbabiltyTensorHistory, ratioActionProbabiltyTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationClipModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not actionNoiseTensor) then actionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanTensor[1]}) end

		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters = ActorModel:getModelParameters(true)

		ActorModel:setModelParameters(NewProximalPolicyOptimizationClipModel.OldActorModelParameters, true)

		local oldPolicyActionMeanTensor = ActorModel:forwardPropagate(previousFeatureTensor)

		NewProximalPolicyOptimizationClipModel.OldActorModelParameters = ActorModel:getModelParameters(true)

		local oldPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(oldPolicyActionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

		local currentPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

		local ratioActionProbabiltyTensor

		if (NewProximalPolicyOptimizationClipModel.useLogProbabilities) then

			ratioActionProbabiltyTensor = AqwamTensorLibrary:applyFunction(math.exp, AqwamTensorLibrary:subtract(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor))

		else

			currentPolicyActionProbabilityTensor = AqwamTensorLibrary:applyFunction(math.exp, currentPolicyActionProbabilityTensor)

			oldPolicyActionProbabilityTensor = AqwamTensorLibrary:applyFunction(math.exp, oldPolicyActionProbabilityTensor)

			ratioActionProbabiltyTensor = AqwamTensorLibrary:divide(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor)

		end

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(ratioActionProbabiltyTensorHistory, ratioActionProbabiltyTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local discountFactor = NewProximalPolicyOptimizationClipModel.discountFactor

		local lambda = NewProximalPolicyOptimizationClipModel.lambda

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

		NewProximalPolicyOptimizationClipModel.OldActorModelParameters = NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters

		ActorModel:setModelParameters(NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters, true)

		local clipRatio = NewProximalPolicyOptimizationClipModel.clipRatio 

		local lowerClipRatioValue = 1 - clipRatio

		local upperClipRatioValue = 1 + clipRatio

		local ratioValueModifierFunction = function(ratioValue) -- This is for the gradient of Proximal Policy Optimization clipped loss.

			return ((ratioValue == 0) and 0) or ((ratioValue >= lowerClipRatioValue) and (ratioValue <= upperClipRatioValue) and ratioValue) or 0

		end

		for h, featureTensor in ipairs(featureTensorHistory) do

			local ratioActionProbabilityTensor = AqwamTensorLibrary:applyFunction(ratioValueModifierFunction, ratioActionProbabiltyTensorHistory[h])

			local actorLossTensor = AqwamTensorLibrary:multiply(advantageValueHistory[h], ratioActionProbabilityTensor)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

			ActorModel:forwardPropagate(featureTensor, true)

			CriticModel:forwardPropagate(featureTensor, true)

			ActorModel:update(actorLossTensor, true)

			CriticModel:update(criticLoss, true)

		end

		NewProximalPolicyOptimizationClipModel.CurrentActorModelParameters = ActorModel:getModelParameters(true)

		table.clear(featureTensorHistory)

		table.clear(ratioActionProbabiltyTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewProximalPolicyOptimizationClipModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(ratioActionProbabiltyTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewProximalPolicyOptimizationClipModel

end

return ProximalPolicyOptimizationClipModel
