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

DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultAveragingRate = 0.995

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function DeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)
	
	NewDeepDoubleStateActionRewardStateActionModel:setName("DeepDoubleStateActionRewardStateAction")

	NewDeepDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepDoubleStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local qTensor = Model:forwardPropagate(currentFeatureTensor)

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

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

		Model:forwardPropagate(previousFeatureTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		local TargetWeightTensorArray = Model:getWeightTensorArray(true)

		TargetWeightTensorArray = rateAverageWeightTensorArray(NewDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		Model:setWeightTensorArray(TargetWeightTensorArray, true)

		return temporalDifferenceErrorTensor

	end)

	NewDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()
		
	end)

	NewDeepDoubleStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()
		
	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

return DeepDoubleStateActionRewardStateActionModel
