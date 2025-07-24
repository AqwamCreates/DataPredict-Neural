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

	local actionProbabilityTensorHistory = {}

	local rewardValueHistory = {}

	local advantageValueHistory = {}

	NewRecurrentVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local CriticModel = NewRecurrentVanillaPolicyGradientModel.CriticModel
		
		local ActorModel = NewRecurrentVanillaPolicyGradientModel.ActorModel
		
		local actorHiddenStateTensor = NewRecurrentVanillaPolicyGradientModel.actorHiddenStateTensor
		
		local criticHiddenStateValue = NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue
		
		if (not actorHiddenStateTensor) then

			local ClassesList = ActorModel:getClassesList()

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

		local logActionProbabilityTensor = AqwamTensorLibrary:logarithm(actionProbabilityTensor)

		local actorLossTensor = AqwamTensorLibrary:multiply(logActionProbabilityTensor, advantageValue)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentVanillaPolicyGradientModel.actorHiddenStateTensor = logActionProbabilityTensor

		NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local CriticModel = NewRecurrentVanillaPolicyGradientModel.CriticModel
		
		local criticHiddenStateValue = NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue or 0

		local actionTensorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(actionMeanTensor, actionTensorPart1)

		local zScoreTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, actionMeanTensor)

		local zScoreTensor = AqwamTensorLibrary:divide(zScoreTensorPart1, actionStandardDeviationTensor)

		local squaredZScoreTensor = AqwamTensorLibrary:power(zScoreTensor, 2)

		local logActionProbabilityTensorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationTensor)

		local logActionProbabilityTensorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityTensorPart1)

		local logActionProbabilityTensorPart3 = AqwamTensorLibrary:add(squaredZScoreTensor, logActionProbabilityTensorPart2)

		local logActionProbabilityTensor = AqwamTensorLibrary:add(logActionProbabilityTensorPart3, math.log(2 * math.pi))

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor, criticHiddenStateValue)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor, previousCriticValue)[1][1]

		local advantageValue = rewardValue + (NewRecurrentVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local actorLossTensor = AqwamTensorLibrary:multiply(logActionProbabilityTensor, advantageValue)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityTensorHistory, logActionProbabilityTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)
		
		NewRecurrentVanillaPolicyGradientModel.actorHiddenStateTensor = logActionProbabilityTensor

		NewRecurrentVanillaPolicyGradientModel.criticHiddenStateValue = previousCriticValue

		return advantageValue

	end)

	NewRecurrentVanillaPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewRecurrentVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewRecurrentVanillaPolicyGradientModel.CriticModel
		
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

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewRecurrentVanillaPolicyGradientModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewRecurrentVanillaPolicyGradientModel

end

return RecurrentVanillaPolicyGradientModel