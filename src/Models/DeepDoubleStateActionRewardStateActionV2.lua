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

local DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local defaultEpsilon = 0.5

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

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepDoubleExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon
		
		local averagingRate = NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate
		
		local EligibilityTrace = NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace
		
		local TargetWeightTensorArray = NewDeepDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray
		
		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		if (not PrimaryWeightTensorArray) then PrimaryWeightTensorArray = Model:generateLayers() end
		
		if (not TargetWeightTensorArray) then TargetWeightTensorArray = PrimaryWeightTensorArray end

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local primaryPreviousActionIndex = table.find(ClassesList, previousAction)

		local primaryPreviousQTensor = Model:forwardPropagate(previousFeatureTensor)
		
		local primaryCurrentQTensor = Model:forwardPropagate(currentFeatureTensor)
		
		local primaryCurrentActionIndex = table.find(ClassesList, currentAction)
		
		local unwrappedPrimaryCurrentTensor = primaryCurrentQTensor[1]
		
		local maximumPrimaryCurrentQValue = unwrappedPrimaryCurrentTensor[primaryCurrentActionIndex]
		
		Model:setWeightTensorArray(TargetWeightTensorArray, true)
		
		local targetCurrentQTensor = Model:forwardPropagate(currentFeatureTensor)

		for i = 1, numberOfClasses do

			if (unwrappedPrimaryCurrentTensor[i] == maximumPrimaryCurrentQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability
		
		local unwrappedTargetCurrentQTensor = targetCurrentQTensor[1]

		local actionProbability
		
		local isGreedy

		for i, targetCurrentQValue in ipairs(unwrappedTargetCurrentQTensor) do

			isGreedy = (unwrappedPrimaryCurrentTensor[i] == maximumPrimaryCurrentQValue)

			actionProbability = (isGreedy and greedyActionProbability) or nonGreedyActionProbability

			expectedQValue = expectedQValue + (targetCurrentQValue * actionProbability)

		end

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local primaryPreviousQValue = primaryPreviousQTensor[1][primaryPreviousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][primaryPreviousActionIndex] = temporalDifferenceError
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(1, primaryPreviousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewDeepDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
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

function DeepDoubleExpectedStateActionRewardStateActionModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:deepCopyTable(TargetWeightTensorArray)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getTargetWeightTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetWeightTensorArray

	else

		return self:deepCopyTable(self.TargetWeightTensorArray)

	end

end

return DeepDoubleExpectedStateActionRewardStateActionModel
