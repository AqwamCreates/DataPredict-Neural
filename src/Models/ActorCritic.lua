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

local ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local function calculateProbability(valueTensor)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local zValueTensor = AqwamTensorLibrary:subtract(valueTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:exponent(zValueTensor)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentTensor)

	local probabilityTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentValue)

	return probabilityTensor

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

function ActorCriticModel.new(parameterDictionary)

	local NewActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewActorCriticModel, ActorCriticModel)

	NewActorCriticModel:setName("ActorCritic")

	local featureTensorHistory = {}

	local actionProbabilityGradientTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	NewActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewActorCriticModel.ActorModel

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor, true)

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local ClassesList = ActorModel:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end

		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel

		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewActorCriticModel.discountFactor)

		for h, featureTensor in ipairs(featureTensorHistory) do

			local criticLoss = criticValueHistory[h] - rewardToGoHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensorHistory[h], criticLoss)

			CriticModel:forwardPropagate(featureTensor, true)

			ActorModel:forwardPropagate(featureTensor, true)

			CriticModel:update(criticLoss, true)

			ActorModel:update(actorLossTensor, true)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	NewActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	return NewActorCriticModel

end

return ActorCriticModel
