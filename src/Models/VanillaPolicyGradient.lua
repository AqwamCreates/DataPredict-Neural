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

local VanillaPolicyGradientModel = {}

VanillaPolicyGradientModel.__index = VanillaPolicyGradientModel

setmetatable(VanillaPolicyGradientModel, DeepReinforcementLearningActorCriticBaseModel)

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

function VanillaPolicyGradientModel.new(parameterDictionary)

	local NewVanillaPolicyGradientModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)

	NewVanillaPolicyGradientModel:setName("VanillaPolicyGradient")

	local featureTensorHistory = {}

	local actionProbabilityGradientTensorHistory = {}

	local rewardValueHistory = {}

	local advantageValueHistory = {}

	NewVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionTensor = ActorModel:forwardPropagate(previousFeatureTensor)

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

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

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureTensor)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureTensor)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		for h, featureTensor in ipairs(featureTensorHistory) do

			local advantageValue = advantageValueHistory[h]

			advantageValue = AqwamTensorLibrary:unaryMinus(advantageValue)

			local actorLossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensorHistory[h], advantageValue)

			CriticModel:forwardPropagate(featureTensor, true)

			ActorModel:forwardPropagate(featureTensor, true)

			CriticModel:update(advantageValue, true)

			ActorModel:update(actorLossTensor, true)

		end

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewVanillaPolicyGradientModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewVanillaPolicyGradientModel

end

return VanillaPolicyGradientModel
