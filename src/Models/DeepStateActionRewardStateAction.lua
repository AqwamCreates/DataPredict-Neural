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

DeepStateActionRewardStateActionModel = {}

DeepStateActionRewardStateActionModel.__index = DeepStateActionRewardStateActionModel

setmetatable(DeepStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

function DeepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepStateActionRewardStateActionModel, DeepStateActionRewardStateActionModel)
	
	NewDeepStateActionRewardStateActionModel:setName("DeepStateActionRewardStateAction")
	
	NewDeepStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewDeepStateActionRewardStateActionModel.EligibilityTrace

		local qTensor = Model:forwardPropagate(currentFeatureTensor)

		local discountedQTensor = AqwamTensorLibrary:multiply(discountFactor, qTensor, (1 - terminalStateValue))

		local targetTensor = AqwamTensorLibrary:add(rewardValue, discountedQTensor)

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor)

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:subtract(targetTensor, previousQTensor)
		
		if (EligibilityTrace) then

			local ClassesList = Model:getClassesList()

			local actionIndex = table.find(ClassesList, action)

			EligibilityTrace:increment(actionIndex, discountFactor, {1, #ClassesList})

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		return temporalDifferenceErrorTensor

	end)

	NewDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepStateActionRewardStateActionModel

end

return DeepStateActionRewardStateActionModel
