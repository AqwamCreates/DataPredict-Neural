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

local RecurrentReinforcementLearningBaseModel = require(script.Parent.RecurrentReinforcementLearningBaseModel)

RecurrentREINFORCEModel = {}

RecurrentREINFORCEModel.__index = RecurrentREINFORCEModel

setmetatable(RecurrentREINFORCEModel, RecurrentReinforcementLearningBaseModel)

local function calculateProbability(valueTensor)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueTensor)

	local subtractedZTensor = AqwamTensorLibrary:subtract(valueTensor, highestActionValue)

	local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

	local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

	local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

	return targetActionTensor

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

function RecurrentREINFORCEModel.new(parameterDictionary)

	local NewRecurrentREINFORCEModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentREINFORCEModel, RecurrentREINFORCEModel)

	NewRecurrentREINFORCEModel:setName("RecurrentREINFORCE")

	local featureTensorArray = {}

	local actionProbabilityTensorArray = {}

	local rewardValueArray = {}

	NewRecurrentREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local Model = NewRecurrentREINFORCEModel.Model
		
		local hiddenStateTensor = NewRecurrentREINFORCEModel.hiddenStateTensor
		
		if (not hiddenStateTensor) then
			
			local ClassesList = Model:getClassesList()

			hiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})
			
		end

		local actionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local actionProbabilityTensor = calculateProbability(actionTensor)

		local logActionProbabilityTensor = AqwamTensorLibrary:logarithm(actionProbabilityTensor)

		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityTensorArray, logActionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)
		
		NewRecurrentREINFORCEModel.hiddenStateTensor = logActionProbabilityTensor

	end)

	NewRecurrentREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensordimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensordimensionSizeArray) 

		end

		local actionTensorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(actionMeanTensor, actionTensorPart1)

		local zScoreTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, actionMeanTensor)

		local zScoreTensor = AqwamTensorLibrary:divide(zScoreTensorPart1, actionStandardDeviationTensor)

		local squaredZScoreTensor = AqwamTensorLibrary:power(zScoreTensor, 2)

		local logActionProbabilityTensorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationTensor)

		local logActionProbabilityTensorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityTensorPart1)

		local logActionProbabilityTensorPart3 = AqwamTensorLibrary:add(squaredZScoreTensor, logActionProbabilityTensorPart2)

		local logActionProbabilityTensor = AqwamTensorLibrary:add(logActionProbabilityTensorPart3, math.log(2 * math.pi))
		
		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityTensorArray, logActionProbabilityTensor)

		table.insert(rewardValueArray, rewardValue)
		
		NewRecurrentREINFORCEModel.hiddenStateTensor = logActionProbabilityTensor

	end)

	NewRecurrentREINFORCEModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewRecurrentREINFORCEModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueArray, NewRecurrentREINFORCEModel.discountFactor)
		
		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionProbabilityTensorArray[1])

		local hiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		for h, actionProbabilityTensor in ipairs(actionProbabilityTensorArray) do
			
			local featureTensor = featureTensorArray[h]

			local lossTensor = AqwamTensorLibrary:multiply(actionProbabilityTensor, rewardToGoArray[h])

			lossTensor = AqwamTensorLibrary:unaryMinus(lossTensor)

			Model:forwardPropagate(featureTensor, hiddenStateTensor)

			Model:update(lossTensor)
			
			hiddenStateTensor = actionProbabilityTensor

		end

		table.clear(featureTensorArray)

		table.clear(actionProbabilityTensorArray)

		table.clear(rewardValueArray)

	end)

	NewRecurrentREINFORCEModel:setResetFunction(function()

		table.clear(featureTensorArray)

		table.clear(actionProbabilityTensorArray)

		table.clear(rewardValueArray)

	end)

	return NewRecurrentREINFORCEModel

end

return RecurrentREINFORCEModel