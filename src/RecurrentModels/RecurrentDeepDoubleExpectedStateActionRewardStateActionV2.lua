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

local RecurrentDeepReinforcementLearningBaseModel = require(script.Parent.RecurrentDeepReinforcementLearningBaseModel)

local RecurrentDeepDoubleExpectedStateActionRewardStateActionModel = {}

RecurrentDeepDoubleExpectedStateActionRewardStateActionModel.__index = RecurrentDeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(RecurrentDeepDoubleExpectedStateActionRewardStateActionModel, RecurrentDeepReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	local NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel = RecurrentDeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel, RecurrentDeepDoubleExpectedStateActionRewardStateActionModel)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = parameterDictionary.primaryHiddenStateTensor

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = parameterDictionary.targetHiddenStateTensor

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local discountFactor = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.discountFactor

		local epsilon = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.epsilon

		local averagingRate = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate

		local EligibilityTrace = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		local TargetWeightTensorArray = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray
		
		local primaryHiddenStateTensor = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor
		
		local targetHiddenStateTensor = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor
		
		local ClassesList = Model:getClassesList()
		
		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not primaryHiddenStateTensor) then primaryHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		if (not targetHiddenStateTensor) then targetHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		if (not PrimaryWeightTensorArray) then PrimaryWeightTensorArray = Model:generateLayers() end

		if (not TargetWeightTensorArray) then TargetWeightTensorArray = PrimaryWeightTensorArray end

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local primaryPreviousActionIndex = table.find(ClassesList, previousAction)

		local primaryPreviousQTensor = Model:forwardPropagate(previousFeatureTensor, primaryHiddenStateTensor)

		local primaryCurrentQTensor = Model:forwardPropagate(currentFeatureTensor, primaryPreviousQTensor)

		local primaryCurrentActionIndex = table.find(ClassesList, currentAction)

		local unwrappedPrimaryCurrentTensor = primaryCurrentQTensor[1]

		local maximumPrimaryCurrentQValue = unwrappedPrimaryCurrentTensor[primaryCurrentActionIndex]

		Model:setWeightTensorArray(TargetWeightTensorArray, true)
		
		local targetPreviousQTensor = Model:forwardPropagate(previousFeatureTensor, targetHiddenStateTensor)

		local targetCurrentQTensor = Model:forwardPropagate(currentFeatureTensor, targetPreviousQTensor)

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

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][primaryPreviousActionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(1, primaryPreviousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-RecurrentDeep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.

		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = primaryPreviousQTensor

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = targetPreviousQTensor
		
		return temporalDifferenceError

	end)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local EligibilityTrace = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = nil

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = nil

	end)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() 

		local EligibilityTrace = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = nil

		NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = nil

	end)

	return NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:RecurrentDeepCopyTable(TargetWeightTensorArray)

	end

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:getTargetWeightTensorArray(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.TargetWeightTensorArray

	else

		return self:RecurrentDeepCopyTable(self.TargetWeightTensorArray)

	end

end

return RecurrentDeepDoubleExpectedStateActionRewardStateActionModel
