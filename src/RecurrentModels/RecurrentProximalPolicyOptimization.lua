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

RecurrentProximalPolicyOptimizationModel = {}

RecurrentProximalPolicyOptimizationModel.__index = RecurrentProximalPolicyOptimizationModel

setmetatable(RecurrentProximalPolicyOptimizationModel, DualRecurrentReinforcementLearningActorCriticBaseModel)

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

function RecurrentProximalPolicyOptimizationModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentProximalPolicyOptimizationModel = DualRecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentProximalPolicyOptimizationModel, RecurrentProximalPolicyOptimizationModel)

	NewRecurrentProximalPolicyOptimizationModel:setName("RecurrentProximalPolicyOptimization")

	NewRecurrentProximalPolicyOptimizationModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewRecurrentProximalPolicyOptimizationModel.useLogProbabilities = NewRecurrentProximalPolicyOptimizationModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray = parameterDictionary.CurrentActorWeightTensorArray

	NewRecurrentProximalPolicyOptimizationModel.OldActorWeightTensorArray = parameterDictionary.OldActorWeightTensorArray

	local featureTensorHistory = {}

	local ratioActionProbabiltyTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	NewRecurrentProximalPolicyOptimizationModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewRecurrentProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewRecurrentProximalPolicyOptimizationModel.CriticModel
		
		local actorHiddenStateTensorArray = NewRecurrentProximalPolicyOptimizationModel.actorHiddenStateTensorArray

		local criticHiddenStateValueArray = NewRecurrentProximalPolicyOptimizationModel.criticHiddenStateValueArray

		local ClassesList = ActorModel:getClassesList()

		local outputDimensionSizeArray = {1, #ClassesList}

		local newActorHiddenStateTensor = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local oldActorHiddenStateTensor = actorHiddenStateTensorArray[2] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		local criticHiddenStateValue = criticHiddenStateValueArray[1] or 0

		NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationModel.OldActorWeightTensorArray, true)

		local oldPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, oldActorHiddenStateTensor)

		NewRecurrentProximalPolicyOptimizationModel.OldActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local oldPolicyActionProbabilityTensor = calculateCategoricalProbability(oldPolicyActionTensor)

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray, true)

		local currentPolicyActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, newActorHiddenStateTensor)

		local currentPolicyActionProbabilityTensor = calculateCategoricalProbability(currentPolicyActionTensor)

		local classIndex = table.find(ClassesList, previousAction)

		local ratioActionProbabiltyTensor = table.create(#ClassesList, 0)

		local ratioActionProbability

		if (NewRecurrentProximalPolicyOptimizationModel.useLogProbabilities) then

			ratioActionProbability = math.exp(math.log(currentPolicyActionProbabilityTensor[1][classIndex]) - math.log(oldPolicyActionProbabilityTensor[1][classIndex]))

		else

			ratioActionProbability = currentPolicyActionProbabilityTensor[1][classIndex] / oldPolicyActionProbabilityTensor[1][classIndex]

		end

		ratioActionProbabiltyTensor[classIndex] = ratioActionProbability

		ratioActionProbabiltyTensor = {ratioActionProbabiltyTensor}

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

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

	NewRecurrentProximalPolicyOptimizationModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local ActorModel = NewRecurrentProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewRecurrentProximalPolicyOptimizationModel.CriticModel
		
		local actorHiddenStateTensorArray = NewRecurrentProximalPolicyOptimizationModel.actorHiddenStateTensorArray
		
		local criticHiddenStateValueArray = NewRecurrentProximalPolicyOptimizationModel.criticHiddenStateValueArray
	
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)
		
		local newActorHiddenStateTensor = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		local oldActorHiddenStateTensor = actorHiddenStateTensorArray[2] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		local criticHiddenStateValue = criticHiddenStateValueArray[1] or 0

		NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationModel.OldActorWeightTensorArray, true)

		local oldPolicyActionMeanTensor = ActorModel:forwardPropagate(previousFeatureTensor, oldActorHiddenStateTensor)

		NewRecurrentProximalPolicyOptimizationModel.OldActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local oldPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(oldPolicyActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local currentPolicyActionProbabilityTensor = calculateDiagonalGaussianProbability(previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local ratioActionProbabiltyTensor

		if (NewRecurrentProximalPolicyOptimizationModel.useLogProbabilities) then

			ratioActionProbabiltyTensor = AqwamTensorLibrary:applyFunction(math.exp, AqwamTensorLibrary:subtract(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor))

		else

			currentPolicyActionProbabilityTensor = AqwamTensorLibrary:applyFunction(math.exp, currentPolicyActionProbabilityTensor)

			oldPolicyActionProbabilityTensor = AqwamTensorLibrary:applyFunction(math.exp, oldPolicyActionProbabilityTensor)

			ratioActionProbabiltyTensor = AqwamTensorLibrary:divide(currentPolicyActionProbabilityTensor, oldPolicyActionProbabilityTensor)

		end

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentProximalPolicyOptimizationModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(ratioActionProbabiltyTensorHistory, ratioActionProbabiltyTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)
		
		actorHiddenStateTensorArray[1] = previousActionMeanTensor
		
		actorHiddenStateTensorArray[2] = oldPolicyActionMeanTensor
		
		criticHiddenStateValueArray[1] = previousCriticValue

		return advantageValue

	end)

	NewRecurrentProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentProximalPolicyOptimizationModel.ActorModel

		local CriticModel = NewRecurrentProximalPolicyOptimizationModel.CriticModel

		local discountFactor = NewRecurrentProximalPolicyOptimizationModel.discountFactor

		local lambda = NewRecurrentProximalPolicyOptimizationModel.lambda

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

		NewRecurrentProximalPolicyOptimizationModel.OldActorWeightTensorArray = NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray

		ActorModel:setWeightTensorArray(NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray, true)

		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(ratioActionProbabiltyTensorHistory[1])

		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		local criticHiddenStateTensor = {{0}}

		for h, featureTensor in ipairs(featureTensorHistory) do

			local ratioActionProbabilityTensor = ratioActionProbabiltyTensorHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(ratioActionProbabilityTensor, advantageValueHistory[h])

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenStateTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenStateTensor)

			ActorModel:update(actorLossTensor)

			CriticModel:update(criticLoss)

		end

		NewRecurrentProximalPolicyOptimizationModel.CurrentActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		table.clear(featureTensorHistory)

		table.clear(ratioActionProbabiltyTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewRecurrentProximalPolicyOptimizationModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(ratioActionProbabiltyTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewRecurrentProximalPolicyOptimizationModel

end

return RecurrentProximalPolicyOptimizationModel
