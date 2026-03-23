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

local defaultEpsilon = 0.5

function DeepDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel:setName("DeepExpectedStateActionRewardStateActionV1")
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewDeepDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or {}

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetTensor = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorTensor, temporalDifferenceError = NewDeepDoubleExpectedStateActionRewardStateActionModel:generateTemporalDifferenceErrorTensor(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:saveWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForUpdate)
		
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

function DeepDoubleExpectedStateActionRewardStateActionModel:saveWeightTensorArrayFromWeightTensorArrayArray(index)

	self.WeightTensorArrayArray[index] = self.Model:getWeightTensorArray()

end

function DeepDoubleExpectedStateActionRewardStateActionModel:loadWeightTensorArrayFromWeightTensorArrayArray(index)

	local Model = self.Model

	local CurrentWeightTensorArray = self.WeightTensorArrayArray[index] or Model:generateLayers()

	Model:setWeightTensorArray(CurrentWeightTensorArray, true)

end

function DeepDoubleExpectedStateActionRewardStateActionModel:generateTemporalDifferenceErrorTensor(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)
	
	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local epsilon = self.epsilon
	
	local EligibilityTrace = self.EligibilityTrace

	local expectedQValue = 0

	local numberOfGreedyActions = 0
	
	local ClassesList = Model:getClassesList()

	local numberOfClasses = #ClassesList

	local actionIndex = table.find(ClassesList, previousAction)
	
	self:loadWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForUpdate)

	local previousQTensor = Model:forwardPropagate(previousFeatureTensor)
	
	self:loadWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForTargetTensor)

	local currentQTensor = Model:forwardPropagate(currentFeatureTensor)

	local maximumCurrentQValue = AqwamTensorLibrary:findMaximumValue(currentQTensor)

	local unwrappedCurrentQTensor = currentQTensor[1]

	for i = 1, numberOfClasses, 1 do

		if (unwrappedCurrentQTensor[i] == maximumCurrentQValue) then

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

	end

	local nonGreedyActionProbability = epsilon / numberOfClasses

	local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability
	
	local actionProbability

	for _, qValue in ipairs(unwrappedCurrentQTensor) do
		
		actionProbability = ((qValue == maximumCurrentQValue) and greedyActionProbability) or nonGreedyActionProbability

		expectedQValue = expectedQValue + (qValue * actionProbability)

	end

	local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)
	
	local previousQValue = previousQTensor[1][actionIndex]

	local temporalDifferenceError = targetQValue - previousQValue
	
	local outputDimensionSizeArray = {1, numberOfClasses}

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)
	
	temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
	
	if (EligibilityTrace) then

		EligibilityTrace:increment(1, actionIndex, discountFactor, outputDimensionSizeArray)

		temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

	end

	return temporalDifferenceErrorTensor, temporalDifferenceError

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setWeightTensorArray1(WeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[1] = WeightTensorArray1

	else

		self.WeightTensorArrayArray[1] = self:deepCopyTable(WeightTensorArray1)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setWeightTensorArray2(WeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[2] = WeightTensorArray2

	else

		self.WeightTensorArrayArray[2] = self:deepCopyTable(WeightTensorArray2)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.WeightTensorArrayArray[1])

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.WeightTensorArrayArray[2])

	end

end

return DeepDoubleExpectedStateActionRewardStateActionModel
