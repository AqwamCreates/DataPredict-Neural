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

RecurrentOffPolicyMonteCarloControlModel = {}

RecurrentOffPolicyMonteCarloControlModel.__index = RecurrentOffPolicyMonteCarloControlModel

setmetatable(RecurrentOffPolicyMonteCarloControlModel, RecurrentReinforcementLearningBaseModel)

local defaultTargetPolicyFunction = "StableSoftmax"

local targetPolicyFunctionList = {

	["Greedy"] = function (actionVector)

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionVector)

		local targetActionTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

		local highestActionValue = -math.huge

		local indexWithHighestActionValue

		for i, actionValue in ipairs(actionVector[1]) do

			if (actionValue > highestActionValue) then

				highestActionValue = actionValue

				indexWithHighestActionValue = i

			end

		end

		targetActionTensor[1][indexWithHighestActionValue] = highestActionValue

		return targetActionTensor

	end,

	["Softmax"] = function (actionVector) -- apparently roblox doesn't really handle very small values such as math.exp(-1000), so I added a more stable computation exp(a) / exp(b) -> exp (a - b)

		local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, actionVector)

		local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

		local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

		return targetActionTensor

	end,

	["StableSoftmax"] = function (actionVector)

		local highestActionValue = AqwamTensorLibrary:findMaximumValue(actionVector)

		local subtractedZVector = AqwamTensorLibrary:subtract(actionVector, highestActionValue)

		local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, subtractedZVector)

		local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

		local targetActionVector = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,

}

function RecurrentOffPolicyMonteCarloControlModel.new(parameterDictionary)

	local NewRecurrentOffPolicyMonteCarloControlModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentOffPolicyMonteCarloControlModel, RecurrentOffPolicyMonteCarloControlModel)

	NewRecurrentOffPolicyMonteCarloControlModel:setName("RecurrentOffPolicyMonteCarloControl")

	NewRecurrentOffPolicyMonteCarloControlModel.targetPolicyFunction = parameterDictionary.targetPolicyFunction or defaultTargetPolicyFunction

	local featureTensorHistory = {}

	local actionTensorHistory = {}

	local rewardValueHistory = {}

	NewRecurrentOffPolicyMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)
		
		local Model = NewRecurrentOffPolicyMonteCarloControlModel.Model

		local hiddenStateTensor = NewRecurrentOffPolicyMonteCarloControlModel.hiddenStateTensor

		if (not hiddenStateTensor) then

			local ClassesList = Model:getClassesList()

			hiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})

		end

		local actionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionTensorHistory, actionTensor)

		table.insert(rewardValueHistory, rewardValue)
		
		NewRecurrentOffPolicyMonteCarloControlModel.hiddenStateTensor = actionTensor

	end)

	NewRecurrentOffPolicyMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewRecurrentOffPolicyMonteCarloControlModel.Model

		local targetPolicyFunction = targetPolicyFunctionList[NewRecurrentOffPolicyMonteCarloControlModel.targetPolicyFunction]

		local discountFactor = NewRecurrentOffPolicyMonteCarloControlModel.discountFactor

		local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionTensorHistory[1])

		local cTensor = AqwamTensorLibrary:createTensor(actionTensorDimensionSizeArray, 0) 

		local weightTensor = AqwamTensorLibrary:createTensor(actionTensorDimensionSizeArray, 1)
		
		local hiddenStateTensor = AqwamTensorLibrary:createTensor(actionTensorDimensionSizeArray)

		local discountedReward = 0

		for h = #actionTensorHistory, 1, -1 do

			discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

			cTensor = AqwamTensorLibrary:add(cTensor, weightTensor)
			
			local featureTensor = featureTensorHistory[h]

			local actionTensor = actionTensorHistory[h]

			local lossTensorPart1 = AqwamTensorLibrary:divide(weightTensor, cTensor)

			local lossTensorPart2 = AqwamTensorLibrary:subtract(discountedReward, actionTensor)

			local lossTensor = AqwamTensorLibrary:multiply(lossTensorPart1, lossTensorPart2, -1) -- The original non-deep off-policy Monte-Carlo Control version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the loss vector by multiplying it with -1 to make the neural network to perform gradient ascent.

			local targetActionTensor = targetPolicyFunction(actionTensor)

			local actionRatioTensor = AqwamTensorLibrary:divide(targetActionTensor, actionTensor)

			weightTensor = AqwamTensorLibrary:multiply(weightTensor, actionRatioTensor)

			Model:forwardPropagate(featureTensor, hiddenStateTensor)

			Model:update(lossTensor)
			
			hiddenStateTensor = actionTensor

		end

		table.clear(featureTensorHistory)

		table.clear(actionTensorHistory)

		table.clear(rewardValueHistory)

	end)

	NewRecurrentOffPolicyMonteCarloControlModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(actionTensorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewRecurrentOffPolicyMonteCarloControlModel

end

function RecurrentOffPolicyMonteCarloControlModel:setParameters(parameterDictionary)

	self.targetPolicyFunction = parameterDictionary.targetPolicyFunction or self.targetPolicyFunction

	self.discountFactor = parameterDictionary.discountFactor or self.discountFactor

end

return RecurrentOffPolicyMonteCarloControlModel