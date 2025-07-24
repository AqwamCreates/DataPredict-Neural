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

RecurrentDeepDoubleExpectedStateActionRewardStateActionModel = {}

RecurrentDeepDoubleExpectedStateActionRewardStateActionModel.__index = RecurrentDeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(RecurrentDeepDoubleExpectedStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultAveragingRate = 0.995

local defaultLambda = 0

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel, RecurrentDeepDoubleExpectedStateActionRewardStateActionModel)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setName("RecurrentDeepDoubleExpectedStateActionRewardStateAction")

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceTensor = parameterDictionary.eligibilityTraceTensor

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local epsilon = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.epsilon

		local lambda = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.lambda

		local discountFactor = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.discountFactor
		
		local hiddenStateTensor = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.hiddenStateTensor
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local actionIndex = table.find(ClassesList, action)

		local previousTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		local targetTensor = Model:forwardPropagate(currentFeatureTensor, previousTensor)

		local maxQValue = targetTensor[1][actionIndex]

		for i = 1, numberOfClasses, 1 do

			if (targetTensor[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(targetTensor[1]) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local lastValue = previousTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError

		if (lambda ~= 0) then

			local eligibilityTraceTensor = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceTensor

			if (not eligibilityTraceTensor) then eligibilityTraceTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

			eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * lambda)

			eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

			temporalDifferenceErrorTensor = AqwamTensorLibrary:multiply(temporalDifferenceErrorTensor, eligibilityTraceTensor)

			NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceTensor = eligibilityTraceTensor

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		local TargetWeightTensorArray = Model:getWeightTensorArray(true)

		TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		Model:setWeightTensorArray(TargetWeightTensorArray, true)

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.hiddenStateTensor = previousTensor

		return temporalDifferenceError

	end)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceTensor = nil

	end)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() 

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceTensor = nil

	end)

	return NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel

end

return RecurrentDeepDoubleExpectedStateActionRewardStateActionModel