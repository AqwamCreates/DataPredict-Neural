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

local AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

setmetatable(AdvantageActorCriticModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultLambda = 0

local function calculateProbability(valueTensor)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local zValueTensor = AqwamTensorLibrary:subtract(valueTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:exponent(zValueTensor)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentTensor)

	local probabilityTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentValue)

	return probabilityTensor

end

function AdvantageActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAdvantageActorCriticModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)

	AdvantageActorCriticModel:setName("AdvantageActorCritic")

	NewAdvantageActorCriticModel.lambda = parameterDictionary.lambda or defaultLambda

	local featureTensorHistory = {}

	local advantageValueHistory = {}

	local actionProbabilityGradientTensorHistory = {}

	NewAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local ClassesList = ActorModel:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end

		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local lambda = NewAdvantageActorCriticModel.lambda

		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			local discountFactor = NewAdvantageActorCriticModel.discountFactor

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue)

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end

		for h, featureTensor in ipairs(featureTensorHistory) do

			local advantageValue = advantageValueHistory[h]

			advantageValue = -advantageValue

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensorHistory[h], advantageValue)

			CriticModel:forwardPropagate(featureTensor, true)

			ActorModel:forwardPropagate(featureTensor, true)

			CriticModel:update(advantageValue, true)

			ActorModel:update(actorLossTensor, true)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(advantageValueHistory)

	end)

	NewAdvantageActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(advantageValueHistory)

	end)

	return NewAdvantageActorCriticModel

end

return AdvantageActorCriticModel
