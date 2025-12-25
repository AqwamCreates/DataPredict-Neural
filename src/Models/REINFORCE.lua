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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

local DeepREINFORCEModel = {}

DeepREINFORCEModel.__index = DeepREINFORCEModel

setmetatable(DeepREINFORCEModel, ReinforcementLearningBaseModel)

local function calculateProbability(valueTensor)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local zValueTensor = AqwamTensorLibrary:subtract(valueTensor, maximumValue)

	local exponentTensor = AqwamTensorLibrary:exponent(zValueTensor)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentTensor)

	local probabilityTensor = AqwamTensorLibrary:divide(exponentTensor, sumExponentValue)

	return probabilityTensor

end

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function DeepREINFORCEModel.new(parameterDictionary)

	local NewDeepREINFORCEModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepREINFORCEModel, DeepREINFORCEModel)

	NewDeepREINFORCEModel:setName("DeepREINFORCE")

	local featureTensorArray = {}

	local actionProbabilityGradientTensorHistory = {}

	local rewardValueHistory = {}

	NewDeepREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewDeepREINFORCEModel.Model

		local actionTensor = Model:forwardPropagate(previousFeatureTensor)

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local ClassesList = Model:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end

		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}

		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewDeepREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)

		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityGradientTensorHistory, actionProbabilityGradientTensor)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewDeepREINFORCEModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewDeepREINFORCEModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewDeepREINFORCEModel.discountFactor)

		for h, actionProbabilityGradientTensor in ipairs(actionProbabilityGradientTensorHistory) do

			local lossTensor = AqwamTensorLibrary:multiply(actionProbabilityGradientTensor, rewardToGoArray[h])

			lossTensor = AqwamTensorLibrary:unaryMinus(lossTensor)

			Model:forwardPropagate(featureTensorArray[h], true)

			Model:update(lossTensor, true)

		end

		table.clear(featureTensorArray)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

	end)

	NewDeepREINFORCEModel:setResetFunction(function()

		table.clear(featureTensorArray)

		table.clear(actionProbabilityGradientTensorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewDeepREINFORCEModel

end

return DeepREINFORCEModel
