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

RecurrentTemporalDifferenceActorCriticModel = {}

RecurrentTemporalDifferenceActorCriticModel.__index = RecurrentTemporalDifferenceActorCriticModel

setmetatable(RecurrentTemporalDifferenceActorCriticModel, RecurrentReinforcementLearningActorCriticBaseModel)

local function calculateProbability(valueTensor)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local zValueTensor = AqwamTensorLibrary:subtract(valueTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:exponent(zValueTensor)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentTensor)

	local probabilityTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentValue)

	return probabilityTensor

end

function RecurrentTemporalDifferenceActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentTemporalDifferenceActorCriticModel = RecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentTemporalDifferenceActorCriticModel, RecurrentTemporalDifferenceActorCriticModel)

	RecurrentTemporalDifferenceActorCriticModel:setName("RecurrentAdvantageActorCritic")

	NewRecurrentTemporalDifferenceActorCriticModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentTemporalDifferenceActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local ActorModel = NewRecurrentTemporalDifferenceActorCriticModel.ActorModel
		
		local CriticModel = NewRecurrentTemporalDifferenceActorCriticModel.CriticModel
		
		local discountFactor = NewRecurrentTemporalDifferenceActorCriticModel.discountFactor
		
		local EligibilityTrace = NewRecurrentTemporalDifferenceActorCriticModel.EligibilityTrace
		
		local ClassesList = ActorModel:getClassesList()
		
		local outputDimensionSizeArray = {1, #ClassesList}
		
		local actorHiddenStateTensor = NewRecurrentTemporalDifferenceActorCriticModel.actorHiddenStateTensor or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenStateValue = NewRecurrentTemporalDifferenceActorCriticModel.criticHiddenStateValue or 0
		
		local criticHiddenStateTensor = {{criticHiddenStateValue}}

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)

		local previousCriticTensor = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateTensor)

		local currentCriticTensor = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticTensor)

		local previousCriticValue = previousCriticTensor[1][1]

		local temporalDifferenceError = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticTensor[1][1]) - previousCriticValue

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end

		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}
		
		local previousActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)
		
		CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateTensor)
		
		if (EligibilityTrace) then

			local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

			temporalDifferenceErrorVector[1][classIndex] = temporalDifferenceError

			EligibilityTrace:increment(1, classIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

			temporalDifferenceError = temporalDifferenceErrorVector[1][classIndex]

		end
		
		local criticLoss = {{-temporalDifferenceError}}
		
		local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensor, criticLoss)

		ActorModel:update(actorLossTensor)

		CriticModel:update(criticLoss)
		
		NewRecurrentTemporalDifferenceActorCriticModel.actorHiddenStateTensor = previousActionTensor

		NewRecurrentTemporalDifferenceActorCriticModel.criticHiddenStateValue = previousCriticValue

		return temporalDifferenceError

	end)

	NewRecurrentTemporalDifferenceActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)
		
		local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)
		
		if (not previousActionNoiseTensor) then

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end
		
		local ActorModel = NewRecurrentTemporalDifferenceActorCriticModel.ActorModel

		local CriticModel = NewRecurrentTemporalDifferenceActorCriticModel.CriticModel
		
		local actorHiddenStateTensor = NewRecurrentTemporalDifferenceActorCriticModel.actorHiddenStateTensor or AqwamTensorLibrary:createTensor(actionTensorDimensionSizeArray, 0)
		
		local criticHiddenStateValue = NewRecurrentTemporalDifferenceActorCriticModel.criticHiddenStateValue or 0
		
		local criticHiddenStateTensor = {{criticHiddenStateValue}}

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		local previousCriticTensor = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateTensor)

		local currentCriticTensor = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticTensor)
		
		local previousCriticValue = previousCriticTensor[1][1]

		local temporalDifferenceError = rewardValue + (NewRecurrentTemporalDifferenceActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticTensor[1][1]) - previousCriticValue
		
		local criticLoss = {{-temporalDifferenceError}}
		
		local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensor, criticLoss)
		
		ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)
		
		CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateTensor)

		ActorModel:update(actorLossTensor)

		CriticModel:update(criticLoss)
		
		NewRecurrentTemporalDifferenceActorCriticModel.actorHiddenStateTensor = previousActionMeanTensor

		NewRecurrentTemporalDifferenceActorCriticModel.criticHiddenStateValue = previousCriticValue

		return temporalDifferenceError

	end)

	NewRecurrentTemporalDifferenceActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local EligibilityTrace = NewRecurrentTemporalDifferenceActorCriticModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end

	end)

	NewRecurrentTemporalDifferenceActorCriticModel:setResetFunction(function()
		
		local EligibilityTrace = NewRecurrentTemporalDifferenceActorCriticModel.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end

	end)

	return NewRecurrentTemporalDifferenceActorCriticModel

end

return RecurrentTemporalDifferenceActorCriticModel
