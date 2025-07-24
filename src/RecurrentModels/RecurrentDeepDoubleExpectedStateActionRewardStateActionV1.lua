--[[

	--------------------------------------------------------------------

	Aqwam's RecurrentDeep Learning Library (DataPredict Neural)

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

local DualRecurrentReinforcementLearningBaseModel = require(script.Parent.DualRecurrentReinforcementLearningBaseModel)

RecurrentDeepDoubleExpectedStateActionRewardStateActionModel = {}

RecurrentDeepDoubleExpectedStateActionRewardStateActionModel.__index = RecurrentDeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(RecurrentDeepDoubleExpectedStateActionRewardStateActionModel, DualRecurrentReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultLambda = 0

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel = DualRecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel, RecurrentDeepDoubleExpectedStateActionRewardStateActionModel)

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setName("RecurrentDeepDoubleExpectedStateActionRewardStateAction")

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or {}

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceTensor = parameterDictionary.eligibilityTraceTensor

	NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.Model
		
		local hiddenStateTensorArray =  NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.hiddenStateTensorArray

		local WeightTensorArrayArray = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel.WeightTensorArrayArray

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetTensor = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorTensor, temporalDifferenceError = NewRecurrentDeepDoubleExpectedStateActionRewardStateActionModel:generateLossTensor(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForUpdate], true)

		local selectedActionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForUpdate])

		Model:update(negatedTemporalDifferenceErrorTensor)

		WeightTensorArrayArray[selectedModelNumberForUpdate] = Model:getWeightTensorArray(true)

		Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForTargetTensor], true)

		local targetActionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForTargetTensor])

		hiddenStateTensorArray[selectedModelNumberForUpdate] = selectedActionTensor

		hiddenStateTensorArray[selectedModelNumberForTargetTensor] = targetActionTensor

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

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:generateLossTensor(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

	local Model = self.Model

	local discountFactor = self.discountFactor

	local epsilon = self.epsilon

	local lambda = self.lambda
	
	local hiddenStateTensorArray = self.hiddenStateTensorArray

	local WeightTensorArrayArray = self.WeightTensorArrayArray
	
	local ClassesList = Model:getClassesList()

	local numberOfClasses = #ClassesList

	local outputDimensionSizeArray = {1, numberOfClasses}
	
	if (not hiddenStateTensorArray[1]) then hiddenStateTensorArray[1] = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

	if (not hiddenStateTensorArray[2]) then hiddenStateTensorArray[2] = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

	if (not WeightTensorArrayArray[1]) then WeightTensorArrayArray[1] = Model:getWeightTensorArray(true) end

	if (not WeightTensorArrayArray[2]) then WeightTensorArrayArray[2] = Model:getWeightTensorArray(true) end

	local expectedQValue = 0

	local numberOfGreedyActions = 0

	local actionIndex = table.find(ClassesList, action)

	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForUpdate], true)
	
	local previousTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForUpdate])

	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForTargetTensor], true)
	
	local previousTargetQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForTargetTensor])

	local targetTensor = Model:forwardPropagate(currentFeatureTensor, previousTargetQTensor)

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

		local eligibilityTraceTensor = self.eligibilityTraceTensor

		if (not eligibilityTraceTensor) then eligibilityTraceTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

		eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * lambda)

		eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

		temporalDifferenceErrorTensor = AqwamTensorLibrary:multiply(temporalDifferenceErrorTensor, eligibilityTraceTensor)

		self.eligibilityTraceTensor = eligibilityTraceTensor

	end

	return temporalDifferenceErrorTensor, temporalDifferenceError

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setWeightTensorArray1(WeightTensorArray1, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.WeightTensorArrayArray[1] = WeightTensorArray1

	else

		self.WeightTensorArrayArray[1] = deepCopyTable(WeightTensorArray1)

	end

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:setWeightTensorArray2(WeightTensorArray2, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.WeightTensorArrayArray[2] = WeightTensorArray2

	else

		self.WeightTensorArrayArray[2] = deepCopyTable(WeightTensorArray2)

	end

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:getWeightTensorArray1(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.WeightTensorArrayArray[1]

	else

		return deepCopyTable(self.WeightTensorArrayArray[1])

	end

end

function RecurrentDeepDoubleExpectedStateActionRewardStateActionModel:getWeightTensorArray2(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.WeightTensorArrayArray[2]

	else

		return deepCopyTable(self.WeightTensorArrayArray[2])

	end

end

return RecurrentDeepDoubleExpectedStateActionRewardStateActionModel