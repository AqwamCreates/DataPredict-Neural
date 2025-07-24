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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

OffPolicyMonteCarloControlModel = {}

OffPolicyMonteCarloControlModel.__index = OffPolicyMonteCarloControlModel

setmetatable(OffPolicyMonteCarloControlModel, ReinforcementLearningBaseModel)

local defaultTargetPolicyFunction = "StableSoftmax"

local targetPolicyFunctionList = {

	["Greedy"] = function (actionTensor)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionTensor)

		local targetActionTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

		local highestActionValue = -math.huge

		local indexWithHighestActionValue

		for i, actionValue in ipairs(actionTensor[1]) do

			if (actionValue > highestActionValue) then

				highestActionValue = actionValue

				indexWithHighestActionValue = i

			end

		end

		targetActionTensor[1][indexWithHighestActionValue] = highestActionValue

		return targetActionTensor

	end,

	["Softmax"] = function (actionTensor) -- apparently Lua doesn't really handle very small values such as math.exp(-1000), so I added a more stable computation exp(a) / exp(b) -> exp (a - b)

		local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, actionTensor)

		local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

		local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

		return targetActionTensor

	end,

	["StableSoftmax"] = function (actionTensor)

		local highestActionValue = AqwamTensorLibrary:findMaximumValue(actionTensor)

		local subtractedZTensor = AqwamTensorLibrary:subtract(actionTensor, highestActionValue)

		local exponentActionTensor = AqwamTensorLibrary:applyFunction(math.exp, subtractedZTensor)

		local exponentActionSumTensor = AqwamTensorLibrary:sum(exponentActionTensor, 2)

		local targetActionTensor = AqwamTensorLibrary:divide(exponentActionTensor, exponentActionSumTensor)

		return targetActionTensor

	end,

}

function OffPolicyMonteCarloControlModel.new(parameterDictionary)

	local NewOffPolicyMonteCarloControlModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewOffPolicyMonteCarloControlModel, OffPolicyMonteCarloControlModel)
	
	NewOffPolicyMonteCarloControlModel:setName("OffPolicyMonteCarloControl")
	
	NewOffPolicyMonteCarloControlModel.targetPolicyFunction = parameterDictionary.targetPolicyFunction or defaultTargetPolicyFunction
	
	local featureTensorHistory = {}

	local actionTensorHistory = {}

	local rewardValueHistory = {}

	NewOffPolicyMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local actionTensor = NewOffPolicyMonteCarloControlModel.Model:forwardPropagate(previousFeatureTensor)
		
		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(actionTensorHistory, actionTensor)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewOffPolicyMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewOffPolicyMonteCarloControlModel.Model

		local targetPolicyFunction = targetPolicyFunctionList[NewOffPolicyMonteCarloControlModel.targetPolicyFunction]

		local discountFactor = NewOffPolicyMonteCarloControlModel.discountFactor
		
		local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionTensorHistory[1])

		local cTensor = AqwamTensorLibrary:createTensor(actionTensorDimensionSizeArray, 0) 

		local weightTensor = AqwamTensorLibrary:createTensor(actionTensorDimensionSizeArray, 1)

		local discountedReward = 0

		for h = #actionTensorHistory, 1, -1 do

			discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

			cTensor = AqwamTensorLibrary:add(cTensor, weightTensor)

			local actionTensor = actionTensorHistory[h]

			local lossTensorPart1 = AqwamTensorLibrary:divide(weightTensor, cTensor)

			local lossTensorPart2 = AqwamTensorLibrary:subtract(discountedReward, actionTensor)

			local lossTensor = AqwamTensorLibrary:multiply(lossTensorPart1, lossTensorPart2, -1) -- The original non-deep off-policy Monte-Carlo Control version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the loss vector by multiplying it with -1 to make the neural network to perform gradient ascent.

			local targetActionTensor = targetPolicyFunction(actionTensor)

			local actionRatioTensor = AqwamTensorLibrary:divide(targetActionTensor, actionTensor)

			weightTensor = AqwamTensorLibrary:multiply(weightTensor, actionRatioTensor)

			Model:forwardPropagate(featureTensorHistory[h], true)

			Model:update(lossTensor, true)

		end
		
		table.clear(featureTensorHistory)

		table.clear(actionTensorHistory)

		table.clear(rewardValueHistory)

	end)

	NewOffPolicyMonteCarloControlModel:setResetFunction(function()
		
		table.clear(featureTensorHistory)

		table.clear(actionTensorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewOffPolicyMonteCarloControlModel

end

function OffPolicyMonteCarloControlModel:setParameters(parameterDictionary)
	
	self.targetPolicyFunction = parameterDictionary.targetPolicyFunction or self.targetPolicyFunction

	self.discountFactor = parameterDictionary.discountFactor or self.discountFactor

end

return OffPolicyMonteCarloControlModel