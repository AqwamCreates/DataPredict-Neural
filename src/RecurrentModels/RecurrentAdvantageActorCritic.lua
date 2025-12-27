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

RecurrentAdvantageActorCriticModel = {}

RecurrentAdvantageActorCriticModel.__index = RecurrentAdvantageActorCriticModel

setmetatable(RecurrentAdvantageActorCriticModel, RecurrentReinforcementLearningActorCriticBaseModel)

local defaultLambda = 0

local function calculateProbability(valueTensor)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local zValueTensor = AqwamTensorLibrary:subtract(valueTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:exponent(zValueTensor)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentTensor)

	local probabilityTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentValue)

	return probabilityTensor

end

function RecurrentAdvantageActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentAdvantageActorCriticModel = RecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentAdvantageActorCriticModel, RecurrentAdvantageActorCriticModel)

	RecurrentAdvantageActorCriticModel:setName("RecurrentAdvantageActorCritic")

	NewRecurrentAdvantageActorCriticModel.lambda = parameterDictionary.lambda or defaultLambda

	local featureTensorHistory = {}

	local advantageValueHistory = {}

	local actionProbabilityGradientTensorHistory = {}

	NewRecurrentAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local ActorModel = NewRecurrentAdvantageActorCriticModel.ActorModel
		
		local CriticModel = NewRecurrentAdvantageActorCriticModel.CriticModel
		
		local actorHiddenStateTensor = NewRecurrentAdvantageActorCriticModel.actorHiddenStateTensor

		local criticHiddenStateValue = NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue or 0
		
		local ClassesList = ActorModel:getClassesList()

		if (not actorHiddenStateTensor) then

			actorHiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})

		end

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local advantageValue = rewardValue + (NewRecurrentAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end

		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentAdvantageActorCriticModel.actorHiddenStateTensor = actionTensor

		NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local CriticModel = NewRecurrentAdvantageActorCriticModel.CriticModel
		
		local criticHiddenStateValue = NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue or 0

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentAdvantageActorCriticModel.actorHiddenStateTensor = previousActionMeanTensor

		NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentAdvantageActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentAdvantageActorCriticModel.ActorModel

		local CriticModel = NewRecurrentAdvantageActorCriticModel.CriticModel

		local lambda = NewRecurrentAdvantageActorCriticModel.lambda

		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			local discountFactor = NewRecurrentAdvantageActorCriticModel.discountFactor

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue)

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionProbabilityGradientTensorHistory[1])
		
		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenTensor = {{0}}

		for h, featureTensor in ipairs(featureTensorHistory) do

			local advantageValue = advantageValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensorHistory[h], advantageValue)

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)
			
			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenTensor)
			
			ActorModel:update(actorLossTensor)
			
			CriticModel:update(advantageValue)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(advantageValueHistory)

	end)

	NewRecurrentAdvantageActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(advantageValueHistory)

	end)

	return NewRecurrentAdvantageActorCriticModel

end

return RecurrentAdvantageActorCriticModel
