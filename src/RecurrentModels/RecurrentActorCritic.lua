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

local RecurrentReinforcementLearningActorCriticBaseModel = require(script.Parent.RecurrentReinforcementLearningActorCriticBaseModel)

RecurrentActorCriticModel = {}

RecurrentActorCriticModel.__index = RecurrentActorCriticModel

setmetatable(RecurrentActorCriticModel, RecurrentReinforcementLearningActorCriticBaseModel)

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

function RecurrentActorCriticModel.new(parameterDictionary)

	local NewRecurrentActorCriticModel = RecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentActorCriticModel, RecurrentActorCriticModel)

	NewRecurrentActorCriticModel:setName("RecurrentActorCritic")

	local featureTensorHistory = {}

	local actionProbabilityTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	NewRecurrentActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local ActorModel = NewRecurrentActorCriticModel.ActorModel
		
		local actorHiddenStateTensor = NewRecurrentActorCriticModel.actorHiddenStateTensor

		local criticHiddenStateValue = NewRecurrentActorCriticModel.criticHiddenStateValue or 0

		if (not actorHiddenStateTensor) then

			local ClassesList = ActorModel:getClassesList()

			actorHiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})

		end

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)

		local criticValue = NewRecurrentActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local logActionProbabilityTensor = AqwamTensorLibrary:logarithm(actionProbabilityTensor)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)
		
		NewRecurrentActorCriticModel.actorHiddenStateTensor = logActionProbabilityTensor

		NewRecurrentActorCriticModel.criticHiddenStateValue = criticValue

	end)

	NewRecurrentActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)
		
		if (not actionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end
		
		local criticHiddenStateValue = NewRecurrentActorCriticModel.criticHiddenStateValue or 0

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

		local criticValue = NewRecurrentActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)
		
		NewRecurrentActorCriticModel.actorHiddenStateTensor = logActionProbabilityTensor

		NewRecurrentActorCriticModel.criticHiddenStateValue = criticValue

	end)

	NewRecurrentActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentActorCriticModel.ActorModel

		local CriticModel = NewRecurrentActorCriticModel.CriticModel

		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewRecurrentActorCriticModel.discountFactor)
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionProbabilityTensorHistory[1])

		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenTensor = {{0}}

		for h, featureTensor in ipairs(featureTensorHistory) do

			local criticLoss = rewardToGoHistory[h] - criticValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityTensorHistory[h], criticLoss)

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)
			
			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenTensor)
			
			ActorModel:update(actorLossTensor)
			
			CriticModel:update(criticLoss)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	NewRecurrentActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	return NewRecurrentActorCriticModel

end

return RecurrentActorCriticModel