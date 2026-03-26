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

local RecurrentDoubleExpectedStateActionRewardStateActionModel = {}

RecurrentDoubleExpectedStateActionRewardStateActionModel.__index = RecurrentDoubleExpectedStateActionRewardStateActionModel

setmetatable(RecurrentDoubleExpectedStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

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

function RecurrentDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	local NewRecurrentDoubleExpectedStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDoubleExpectedStateActionRewardStateActionModel, RecurrentDoubleExpectedStateActionRewardStateActionModel)

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewRecurrentDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = parameterDictionary.primaryHiddenStateTensor

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = parameterDictionary.targetHiddenStateTensor

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.Model

		local discountFactor = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.discountFactor

		local epsilon = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.epsilon

		local averagingRate = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.averagingRate

		local EligibilityTrace = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		local TargetWeightTensorArray = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray
		
		local primaryHiddenStateTensor = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor
		
		local targetHiddenStateTensor = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor
		
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

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-Recurrent expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.

		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDoubleExpectedStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = primaryPreviousQTensor

		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = targetPreviousQTensor
		
		return temporalDifferenceError

	end)

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local EligibilityTrace = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = nil

		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = nil

	end)

	NewRecurrentDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() 

		local EligibilityTrace = NewRecurrentDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.primaryHiddenStateTensor = nil

		NewRecurrentDoubleExpectedStateActionRewardStateActionModel.targetHiddenStateTensor = nil

	end)

	return NewRecurrentDoubleExpectedStateActionRewardStateActionModel

end

function RecurrentDoubleExpectedStateActionRewardStateActionModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotRecurrentCopy)

	if (doNotRecurrentCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:RecurrentCopyTable(TargetWeightTensorArray)

	end

end

function RecurrentDoubleExpectedStateActionRewardStateActionModel:getTargetWeightTensorArray(doNotRecurrentCopy)

	if (doNotRecurrentCopy) then

		return self.TargetWeightTensorArray

	else

		return self:RecurrentCopyTable(self.TargetWeightTensorArray)

	end

end

return RecurrentDoubleExpectedStateActionRewardStateActionModel
