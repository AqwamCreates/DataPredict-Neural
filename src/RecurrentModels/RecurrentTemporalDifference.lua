--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local RecurrentReinforcementLearningBaseModel = require(script.Parent.RecurrentReinforcementLearningBaseModel)

RecurrentTemporalDifferenceModel = {}

RecurrentTemporalDifferenceModel.__index = RecurrentTemporalDifferenceModel

setmetatable(RecurrentTemporalDifferenceModel, RecurrentReinforcementLearningBaseModel)

function RecurrentTemporalDifferenceModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentTemporalDifferenceModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentTemporalDifferenceModel, RecurrentTemporalDifferenceModel)

	NewRecurrentTemporalDifferenceModel:setName("RecurrentTemporalDifference")

	NewRecurrentTemporalDifferenceModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentTemporalDifferenceModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewRecurrentTemporalDifferenceModel.Model

		local discountFactor = NewRecurrentTemporalDifferenceModel.discountFactor
		
		local EligibilityTrace = NewRecurrentTemporalDifferenceModel.EligibilityTrace

		local hiddenStateTensor = NewRecurrentTemporalDifferenceModel.hiddenStateTensor

		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor({{1, 1}}) end

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)
		
		local _, maxQValue = Model:predict(currentFeatureTensor, previousQTensor)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

		local temporalDifferenceError = targetValue - previousQTensor[1][1]

		local negatedTemporalDifferenceErrorTensor = {{-temporalDifferenceError}}

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		NewRecurrentTemporalDifferenceModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceError

	end)

	NewRecurrentTemporalDifferenceModel:setEpisodeUpdateFunction(function(terminalStateValue) end)

	NewRecurrentTemporalDifferenceModel:setResetFunction(function() end)

	return NewRecurrentTemporalDifferenceModel

end

return RecurrentTemporalDifferenceModel
