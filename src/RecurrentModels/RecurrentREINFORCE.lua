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

	NewRecurrentREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewRecurrentREINFORCEModel.Model
		
		local hiddenStateTensor = NewRecurrentREINFORCEModel.hiddenStateTensor
		
		local ClassesList = Model:getClassesList()
		
		if (not hiddenStateTensor) then
			
			hiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})
			
		end

		local actionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)
		
		local actionProbabilityTensor = calculateProbability(actionTensor)

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientTensor = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientTensor[i] = (((i == classIndex) and 1) or 0) - actionProbabilityTensor[1][i]

		end
		
		actionProbabilityGradientTensor = {actionProbabilityGradientTensor}

		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityTensorArray, actionProbabilityGradientTensor)

		table.insert(rewardValueArray, rewardValue)
		
		NewRecurrentREINFORCEModel.hiddenStateTensor = actionTensor

	end)

	NewRecurrentREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local actionTensorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		local actionTensor = AqwamTensorLibrary:add(previousActionMeanTensor, actionTensorPart1)

		local actionProbabilityGradientTensorPart1 = AqwamTensorLibrary:subtract(actionTensor, previousActionMeanTensor)

		local actionProbabilityGradientTensorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationTensor, 2)

		local actionProbabilityGradientTensor = AqwamTensorLibrary:divide(actionProbabilityGradientTensorPart1, actionProbabilityGradientTensorPart2)
		
		table.insert(featureTensorArray, previousFeatureTensor)

		table.insert(actionProbabilityTensorArray, actionProbabilityGradientTensor)

		table.insert(rewardValueArray, rewardValue)
		
		NewRecurrentREINFORCEModel.hiddenStateTensor = actionTensor

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
