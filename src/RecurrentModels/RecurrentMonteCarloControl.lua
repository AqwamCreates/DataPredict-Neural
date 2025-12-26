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

RecurrentMonteCarloControlModel = {}

RecurrentMonteCarloControlModel.__index = RecurrentMonteCarloControlModel

setmetatable(RecurrentMonteCarloControlModel, RecurrentReinforcementLearningBaseModel)

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function RecurrentMonteCarloControlModel.new(parameterDictionary)

	local NewRecurrentMonteCarloControlModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentMonteCarloControlModel, RecurrentMonteCarloControlModel)

	NewRecurrentMonteCarloControlModel:setName("RecurrentMonteCarloControl")

	local featureTensorHistory = {}

	local rewardValueHistory = {}

	NewRecurrentMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		table.insert(featureTensorHistory, previousFeatureTensor)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewRecurrentMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewRecurrentMonteCarloControlModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewRecurrentMonteCarloControlModel.discountFactor)
		
		local ClassesList = Model:getClassesList()

		local hiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList})

		for h, featureTensor in ipairs(featureTensorHistory) do

			local averageRewardToGo = rewardToGoArray[h] / h

			local actionTensor = Model:forwardPropagate(featureTensor, hiddenStateTensor)

			Model:update(averageRewardToGo)
			
			hiddenStateTensor = actionTensor

		end

		table.clear(featureTensorHistory)

		table.clear(rewardValueHistory)

	end)

	NewRecurrentMonteCarloControlModel:setResetFunction(function()

		table.clear(featureTensorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewRecurrentMonteCarloControlModel

end

function RecurrentMonteCarloControlModel:setParameters(parameterDictionary)

	self.discountFactor = parameterDictionary.discountFactor or self.discountFactor

end

return RecurrentMonteCarloControlModel
