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

RecurrentVanillaPolicyGradientModel = {}

RecurrentVanillaPolicyGradientModel.__index = RecurrentVanillaPolicyGradientModel

setmetatable(RecurrentVanillaPolicyGradientModel, RecurrentReinforcementLearningActorCriticBaseModel)

local function calculateProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

	return targetActionTensor

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

function RecurrentVanillaPolicyGradientModel.new(parameterDictionary)

	local NewRecurrentVanillaPolicyGradientModel = RecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentVanillaPolicyGradientModel, RecurrentVanillaPolicyGradientModel)

	NewRecurrentVanillaPolicyGradientModel:setName("RecurrentVanillaPolicyGradient")

	local featureTensorHistory = {}

	local actionProbabilityGradientTensorHistory = {}

	local rewardValueHistory = {}

	local advantageValueHistory = {}

	NewRecurrentVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local CriticModel = NewRecurrentVanillaPolicyGradientModel.CriticModel
		
		local ActorModel = NewRecurrentVanillaPolicyGradientModel.ActorModel
		
		local actorHiddenStateTensor = NewRecurrentVanillaPolicyGradientModel.actorHiddenStateTensor
		
		local criticHiddenStateValue = NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue
		
		local ClassesList = ActorModel:getClassesList()
		
		if (not actorHiddenStateTensor) then

			actorHiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})

		end
		
		if (not criticHiddenStateValue) then
			
			criticHiddenStateValue = 0
			
		end

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end

		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentVanillaPolicyGradientModel.actorHiddenStateTensor = actionTensor

		NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local CriticModel = NewRecurrentVanillaPolicyGradientModel.CriticModel
		
		local criticHiddenStateValue = NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue or 0

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentVanillaPolicyGradientModel.actorHiddenStateTensor = previousActionMeanTensor

		NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentVanillaPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewRecurrentVanillaPolicyGradientModel.CriticModel
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionProbabilityGradientTensorHistory[1])
		
		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		local criticHiddenTensor = {{0}}

		for h, featureTensor in ipairs(featureTensorHistory) do

			local advantageValue = -advantageValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensorHistory[h], advantageValue)
			
			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenTensor)

			ActorModel:update(actorLossTensor)

			CriticModel:update(advantageValue)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewRecurrentVanillaPolicyGradientModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewRecurrentVanillaPolicyGradientModel

end

return RecurrentVanillaPolicyGradientModel
