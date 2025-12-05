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

	local actionProbabilityTensorHistory = {}

	local rewardValueHistory = {}
	
	local criticValueHistory = {}

	NewActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local actionTensor = NewActorCriticModel.ActorModel:forwardPropagate(previousFeatureTensor, true)
		
		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local logActionProbabilityTensor = AqwamTensorLibrary:logarithm(actionProbabilityTensor)
		
		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(rewardValueHistory, rewardValue)
		
		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then
			
			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)
			
			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 
			
		end

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local zScoreTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local zScoreTensor = AqwamTensorLibrary:divide(zScoreTensorPart1, previousActionStandardDeviationTensor)

		local squaredZScoreTensor = AqwamTensorLibrary:power(zScoreTensor, 2)

		local logActionProbabilityTensorPart1 = AqwamTensorLibrary:logarithm(previousActionStandardDeviationTensor)

		local logActionProbabilityTensorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityTensorPart1)

		local logActionProbabilityTensorPart3 = AqwamTensorLibrary:add(squaredZScoreTensor, logActionProbabilityTensorPart2)

		local logActionProbabilityTensorPart4 = AqwamTensorLibrary:add(logActionProbabilityTensorPart3, math.log(2 * math.pi))

		local logActionProbabilityTensor = AqwamTensorLibrary:multiply(-0.5, logActionProbabilityTensorPart4)

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel

		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewActorCriticModel.discountFactor)

		for h, featureTensor in ipairs(featureTensorHistory) do

			local criticLoss = rewardToGoHistory[h] - criticValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityTensorHistory[h], criticLoss)
			
			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)

			ActorModel:forwardPropagate(featureTensor, true)

			CriticModel:forwardPropagate(featureTensor, true)
			
			ActorModel:update(actorLossTensor, true)
			
			CriticModel:update(criticLoss, true)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	NewActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	return NewActorCriticModel

end

return ActorCriticModel
