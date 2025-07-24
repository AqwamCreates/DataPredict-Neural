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

local defaultLambda = 0 

function RecurrentDeepStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepStateActionRewardStateActionModel, RecurrentDeepStateActionRewardStateActionModel)

	NewRecurrentDeepStateActionRewardStateActionModel:setName("RecurrentDeepStateActionRewardStateAction")

	NewRecurrentDeepStateActionRewardStateActionModel.lambda = NewRecurrentDeepStateActionRewardStateActionModel.lambda or defaultLambda

	NewRecurrentDeepStateActionRewardStateActionModel.eligibilityTraceTensor = NewRecurrentDeepStateActionRewardStateActionModel.eligibilityTraceTensor

	NewRecurrentDeepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepStateActionRewardStateActionModel.Model

		local lambda = NewRecurrentDeepStateActionRewardStateActionModel.lambda

		local discountFactor = NewRecurrentDeepStateActionRewardStateActionModel.discountFactor
		
		local hiddenStateTensor = NewRecurrentDeepStateActionRewardStateActionModel.hiddenStateTensor
		
		local ClassesList = Model:getClassesList()
		
		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor({1, #ClassesList}) end
		
		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local qTensor = Model:forwardPropagate(currentFeatureTensor, previousQTensor)

		local discountedQTensor = AqwamTensorLibrary:multiply(discountFactor, qTensor, (1 - terminalStateValue))

		local targetTensor = AqwamTensorLibrary:add(rewardValue, discountedQTensor)

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:subtract(targetTensor, previousQTensor)

		if (lambda ~= 0) then

			local actionIndex = table.find(ClassesList, action)

			local eligibilityTraceTensor = NewRecurrentDeepStateActionRewardStateActionModel.eligibilityTraceTensor

			if (not eligibilityTraceTensor) then eligibilityTraceTensor = AqwamTensorLibrary:createTensor({1, #ClassesList}, 0) end

			eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * lambda)

			eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

			temporalDifferenceErrorTensor = AqwamTensorLibrary:multiply(temporalDifferenceErrorTensor, eligibilityTraceTensor)

			NewRecurrentDeepStateActionRewardStateActionModel.eligibilityTraceTensor = eligibilityTraceTensor

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)
		
		NewRecurrentDeepStateActionRewardStateActionModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceErrorTensor

	end)

	NewRecurrentDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		NewRecurrentDeepStateActionRewardStateActionModel.eligibilityTraceTensor = nil
		
	end)

	NewRecurrentDeepStateActionRewardStateActionModel:setResetFunction(function() 

		NewRecurrentDeepStateActionRewardStateActionModel.eligibilityTraceTensor = nil

	end)

	return NewRecurrentDeepStateActionRewardStateActionModel

end

return RecurrentDeepStateActionRewardStateActionModel