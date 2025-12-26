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

	local actionProbabilityGradientTensorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	NewRecurrentActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
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
		
		NewRecurrentActorCriticModel.actorHiddenStateTensor = actionTensor

		NewRecurrentActorCriticModel.criticHiddenStateValue = criticValue

	end)

	NewRecurrentActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)
		
		if (not previousActionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end
		
		local criticHiddenStateValue = NewRecurrentActorCriticModel.criticHiddenStateValue or 0

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		local criticValue = NewRecurrentActorCriticModel.CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)
		
		NewRecurrentActorCriticModel.actorHiddenStateTensor = actionTensor

		NewRecurrentActorCriticModel.criticHiddenStateValue = criticValue

	end)

	NewRecurrentActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentActorCriticModel.ActorModel

		local CriticModel = NewRecurrentActorCriticModel.CriticModel

		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewRecurrentActorCriticModel.discountFactor)
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionProbabilityGradientTensorHistory[1])

		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenTensor = {{0}}

		for h, featureTensor in ipairs(featureTensorHistory) do

			local criticLoss = rewardToGoHistory[h] - criticValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensorHistory[h], criticLoss)

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)
			
			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenTensor)
			
			ActorModel:update(actorLossTensor)
			
			CriticModel:update(criticLoss)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	NewRecurrentActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	return NewRecurrentActorCriticModel

end

return RecurrentActorCriticModel
