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

MonteCarloControlModel = {}

MonteCarloControlModel.__index = MonteCarloControlModel

setmetatable(MonteCarloControlModel, ReinforcementLearningBaseModel)

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function MonteCarloControlModel.new(parameterDictionary)

	local NewMonteCarloControlModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewMonteCarloControlModel, MonteCarloControlModel)
	
	NewMonteCarloControlModel:setName("MonteCarloControl")

	local featureTensorHistory = {}

	local rewardValueHistory = {}

	NewMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewMonteCarloControlModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewMonteCarloControlModel.discountFactor)

		for h, featureTensor in ipairs(featureTensorHistory) do

			local averageRewardToGo = rewardToGoArray[h] / h

			Model:forwardPropagate(featureTensor, true)

			Model:update(averageRewardToGo, true)

		end

		table.clear(featureTensorHistory)

		table.clear(rewardValueHistory)

	end)

	NewMonteCarloControlModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewMonteCarloControlModel

end

function MonteCarloControlModel:setParameters(parameterDictionary)

	self.discountFactor = parameterDictionary.discountFactor or self.discountFactor

end

return MonteCarloControlModel
