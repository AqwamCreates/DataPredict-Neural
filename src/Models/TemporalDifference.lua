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

local TemporalDifferenceModel = {}

TemporalDifferenceModel.__index = TemporalDifferenceModel

setmetatable(TemporalDifferenceModel, ReinforcementLearningBaseModel)

function TemporalDifferenceModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTemporalDifferenceModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTemporalDifferenceModel, TemporalDifferenceModel)
	
	NewTemporalDifferenceModel:setName("DeepTemporalDifference")
	
	NewTemporalDifferenceModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewTemporalDifferenceModel.Model
		
		local discountFactor = NewTemporalDifferenceModel.discountFactor

		local currentQTensor = Model:forwardPropagate(currentFeatureTensor)

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor)
		
		local targetQValue = rewardValue + (discountFactor * currentQTensor[1][1] * (1 - terminalStateValue))
		
		local previousQValue = previousQTensor[1][1]
		
		local temporalDifferenceError = targetQValue - previousQValue
		
		local negatedTemporalDifferenceErrorTensor = {{-temporalDifferenceError}}
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		return temporalDifferenceError

	end)
	
	NewTemporalDifferenceModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
	end)
	
	NewTemporalDifferenceModel:setResetFunction(function() 
		
	end)

	return NewTemporalDifferenceModel

end

function TemporalDifferenceModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return TemporalDifferenceModel
