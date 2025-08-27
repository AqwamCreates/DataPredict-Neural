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

RecurrentDeepStateActionRewardStateActionModel = {}

RecurrentDeepStateActionRewardStateActionModel.__index = RecurrentDeepStateActionRewardStateActionModel

setmetatable(RecurrentDeepStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

function RecurrentDeepStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepStateActionRewardStateActionModel, RecurrentDeepStateActionRewardStateActionModel)

	NewRecurrentDeepStateActionRewardStateActionModel:setName("RecurrentDeepStateActionRewardStateAction")

	NewRecurrentDeepStateActionRewardStateActionModel.EligibilityTrace = NewRecurrentDeepStateActionRewardStateActionModel.EligibilityTrace

	NewRecurrentDeepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepStateActionRewardStateActionModel.Model

		local discountFactor = NewRecurrentDeepStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewRecurrentDeepStateActionRewardStateActionModel.EligibilityTrace
		
		local hiddenStateTensor = NewRecurrentDeepStateActionRewardStateActionModel.hiddenStateTensor
		
		local ClassesList = Model:getClassesList()
		
		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList}) end
		
		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local qTensor = Model:forwardPropagate(currentFeatureTensor, previousQTensor)

		local discountedQTensor = AqwamTensorLibrary:multiply(discountFactor, qTensor, (1 - terminalStateValue))

		local targetTensor = AqwamTensorLibrary:add(rewardValue, discountedQTensor)

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:subtract(targetTensor, previousQTensor)

		if (EligibilityTrace) then

			local ClassesList = Model:getClassesList()

			local actionIndex = table.find(ClassesList, action)

			EligibilityTrace:increment(actionIndex, discountFactor, {1, #ClassesList})

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)
		
		NewRecurrentDeepStateActionRewardStateActionModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceErrorTensor

	end)

	NewRecurrentDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		NewRecurrentDeepStateActionRewardStateActionModel.EligibilityTrace:reset()
		
	end)

	NewRecurrentDeepStateActionRewardStateActionModel:setResetFunction(function() 

		NewRecurrentDeepStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	return NewRecurrentDeepStateActionRewardStateActionModel

end

return RecurrentDeepStateActionRewardStateActionModel
