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

DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultAveragingRate = 0.01

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do
		
		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, PrimaryWeightTensorArray[layer])

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, TargetWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(PrimaryWeightTensorArrayPart, TargetWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function DeepDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel:setName("DeepDoubleExpectedStateActionRewardStateAction")

	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local epsilon = NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon

		local discountFactor = NewDeepDoubleExpectedStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local actionIndex = table.find(ClassesList, action)

		local previousTensor = Model:forwardPropagate(previousFeatureTensor)

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		local targetTensor = Model:forwardPropagate(currentFeatureTensor)

		local maxQValue = AqwamTensorLibrary:findMaximumValue(targetTensor)
		
		local unwrappedTargetTensor = targetTensor[1]

		for i = 1, numberOfClasses, 1 do

			if (unwrappedTargetTensor[i] == maxQValue) then
				
				numberOfGreedyActions = numberOfGreedyActions + 1
				
			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(unwrappedTargetTensor) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local lastValue = previousTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		local TargetWeightTensorArray = Model:getWeightTensorArray(true)

		TargetWeightTensorArray = rateAverageWeightTensorArray(NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		Model:setWeightTensorArray(TargetWeightTensorArray, true)

		return temporalDifferenceError

	end)

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepDoubleExpectedStateActionRewardStateActionModel

end

return DeepDoubleExpectedStateActionRewardStateActionModel
