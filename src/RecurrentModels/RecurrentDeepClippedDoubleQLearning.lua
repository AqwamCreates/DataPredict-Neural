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

	local actionProbabilityTensorHistory = {}

	NewRecurrentAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local ActorModel = NewRecurrentAdvantageActorCriticModel.ActorModel
		
		local CriticModel = NewRecurrentAdvantageActorCriticModel.CriticModel
		
		local actorHiddenStateTensor = NewRecurrentAdvantageActorCriticModel.actorHiddenStateTensor

		local criticHiddenStateValue = NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue or 0

		if (not actorHiddenStateTensor) then

			local ClassesList = ActorModel:getClassesList()

			actorHiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})

		end

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local advantageValue = rewardValue + (NewRecurrentAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local logActionProbabilityTensor = AqwamTensorLibrary:logarithm(actionProbabilityTensor)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentAdvantageActorCriticModel.actorHiddenStateTensor = logActionProbabilityTensor

		NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local CriticModel = NewRecurrentAdvantageActorCriticModel.CriticModel
		
		local criticHiddenStateValue = NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue or 0

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

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentAdvantageActorCriticModel.actorHiddenStateTensor = logActionProbabilityTensor

		NewRecurrentAdvantageActorCriticModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

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
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionProbabilityTensorHistory[1])
		
		local actorHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local criticHiddenTensor = {{0}}

		for h, featureTensor in ipairs(featureTensorHistory) do

			local advantageValue = advantageValueHistory[h]

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityTensorHistory[h], advantageValue)

			actorLossTensor = AqwamTensorLibrary:unaryMinus(actorLossTensor)
			
			actorHiddenStateTensor = ActorModel:forwardPropagate(featureTensor, actorHiddenStateTensor)

			criticHiddenTensor = CriticModel:forwardPropagate(featureTensor, criticHiddenTensor)
			
			ActorModel:update(actorLossTensor)
			
			CriticModel:update(advantageValue)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(advantageValueHistory)

	end)

	NewRecurrentAdvantageActorCriticModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(advantageValueHistory)

	end)

	return NewRecurrentAdvantageActorCriticModel

end

return RecurrentAdvantageActorCriticModel