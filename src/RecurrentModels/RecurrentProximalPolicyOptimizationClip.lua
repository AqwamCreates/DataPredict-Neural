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

RecurrentProximalPolicyOptimizationClipModel = {}

RecurrentProximalPolicyOptimizationClipModel.__index = RecurrentProximalPolicyOptimizationClipModel

setmetatable(RecurrentProximalPolicyOptimizationClipModel, DualRecurrentReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local defaultLambda = 0

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

function RecurrentProximalPolicyOptimizationClipModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentProximalPolicyOptimizationClipModel = DualRecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentProximalPolicyOptimizationClipModel, RecurrentProximalPolicyOptimizationClipModel)

	NewRecurrentProximalPolicyOptimizationClipModel:setName("RecurrentProximalPolicyOptimizationClip")

	NewRecurrentProximalPolicyOptimizationClipModel.clipRatio = parameterDictionary.clipRatio or defaultClipRatio

	NewRecurrentProximalPolicyOptimizationClipModel.lambda = parameterDictionary.lambda or defaultLambda

	NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray = parameterDictionary.CurrentActorWeightTensorArray

	NewRecurrentProximalPolicyOptimizationClipModel.OldActorWeightTensorArray = parameterDictionary.OldActorWeightTensorArray

	local featureTensorHistory = {}

	local ratioActionProbabiltyTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewRecurrentProximalPolicyOptimizationClipModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local ActorModel = NewRecurrentProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewRecurrentProximalPolicyOptimizationClipModel.CriticModel
		
		local actorHiddenStateTensorArray = NewRecurrentProximalPolicyOptimizationClipModel.actorHiddenStateTensorArray

		local criticHiddenStateValueArray = NewRecurrentProximalPolicyOptimizationClipModel.criticHiddenStateValueArray

		local ClassesList = ActorModel:getClassesList()

		local outputDimensionSizeArray = {1, #ClassesList}

		local newActorHiddenStateTensor = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local oldActorHiddenStateTensor = actorHiddenStateTensorArray[2] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenStateValue = criticHiddenStateValueArray[1] or 0

		NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationClipModel.OldActorWeightTensorArray, true)

		local oldPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, oldActorHiddenStateTensor)

		NewRecurrentProximalPolicyOptimizationClipModel.OldActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local oldPolicyActionProbabilityTensor = calculateCategoricalProbability(oldPolicyActionTensor)

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray, true)

		local currentPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, newActorHiddenStateTensor)

		local currentPolicyActionProbabilityTensor = calculateCategoricalProbability(currentPolicyActionTensor)

		local ratioActionProbabiltyTensor = AqwamTensorLibrary:divide(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentProximalPolicyOptimizationClipModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(ratioActionProbabiltyTensorHistory, ratioActionProbabiltyTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)
		
		actorHiddenStateTensorArray[1] = currentPolicyActionTensor

		actorHiddenStateTensorArray[2] = oldPolicyActionTensor

		criticHiddenStateValueArray[1] = previousCriticValue

		return advantageValue

	end)

	NewRecurrentProximalPolicyOptimizationClipModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local ActorModel = NewRecurrentProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewRecurrentProximalPolicyOptimizationClipModel.CriticModel
		
		local actorHiddenStateTensorArray = NewRecurrentProximalPolicyOptimizationClipModel.actorHiddenStateTensorArray

		local criticHiddenStateValueArray = NewRecurrentProximalPolicyOptimizationClipModel.criticHiddenStateValueArray

		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

		local newActorHiddenStateTensor = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local oldActorHiddenStateTensor = actorHiddenStateTensorArray[2] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenStateValue = criticHiddenStateValueArray[1] or 0

		NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationClipModel.OldActorWeightTensorArray, true)

		local oldPolicyActionMeanTensor = ActorModel:forwardPropagate(previousFeatureTensor, oldActorHiddenStateTensor)

		NewRecurrentProximalPolicyOptimizationClipModel.OldActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local oldPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(oldPolicyActionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

		local currentPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor)

		local ratioActionProbabiltyTensor = AqwamTensorLibrary:divide(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, {{criticHiddenStateValue}})[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, {{previousCriticValue}})[1][1]

		local advantageValue = rewardValue + (NewRecurrentProximalPolicyOptimizationClipModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(ratioActionProbabiltyTensorHistory, ratioActionProbabiltyTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)
		
		actorHiddenStateTensorArray[1] = actionMeanTensor

		actorHiddenStateTensorArray[2] = oldPolicyActionMeanTensor

		criticHiddenStateValueArray[1] = previousCriticValue

		return advantageValue

	end)

	NewRecurrentProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewRecurrentProximalPolicyOptimizationClipModel.CriticModel

		local discountFactor = NewRecurrentProximalPolicyOptimizationClipModel.discountFactor

		local lambda = NewRecurrentProximalPolicyOptimizationClipModel.lambda

		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue) -- Insert at the beginning to maintain order

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, discountFactor)

		NewRecurrentProximalPolicyOptimizationClipModel.OldActorWeightTensorArray = NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray, true)
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(ratioActionProbabiltyTensorHistory[1])

		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenStateTensor = {{0}}
		
		local clipRatio = NewRecurrentProximalPolicyOptimizationClipModel.clipRatio 

		local lowerClipRatioValue = 1 - clipRatio

		local upperClipRatioValue = 1 + clipRatio

		local ratioValueModifierFunction = function(ratioValue) -- This is for the gradient of Proximal Policy Optimization clipped loss.

			return ((ratioValue >= lowerClipRatioValue) and (ratioValue <= upperClipRatioValue) and ratioValue) or 0

		end

		for h, featureTensor in ipairs(featureTensorHistory) do

			local ratioActionProbabilityTensor = AqwamTensorLibrary:applyFunction(ratioValueModifierFunction, ratioActionProbabiltyTensorHistory[h])

			local actorLossTensor = AqwamTensorLibrary:multiply(advantageValueHistory[h], ratioActionProbabilityTensor)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenStateTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenStateTensor)

			ActorModel:update(actorLossTensor)

			CriticModel:update(criticLoss)

		end

		NewRecurrentProximalPolicyOptimizationClipModel.CurrentActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		table.clear(featureTensorHistory)

		table.clear(ratioActionProbabiltyTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewRecurrentProximalPolicyOptimizationClipModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(ratioActionProbabiltyTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewRecurrentProximalPolicyOptimizationClipModel

end

return RecurrentProximalPolicyOptimizationClipModel