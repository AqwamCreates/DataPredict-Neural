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

local DualRecurrentReinforcementLearningBaseModel = require(script.Parent.DualRecurrentReinforcementLearningBaseModel)

RecurrentDeepDoubleStateActionRewardStateActionModel = {}

RecurrentDeepDoubleStateActionRewardStateActionModel.__index = RecurrentDeepDoubleStateActionRewardStateActionModel

setmetatable(RecurrentDeepDoubleStateActionRewardStateActionModel, DualRecurrentReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleStateActionRewardStateActionModel = DualRecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleStateActionRewardStateActionModel, RecurrentDeepDoubleStateActionRewardStateActionModel)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setName("RecurrentDeepDoubleStateActionRewardStateAction")

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or {}

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepDoubleStateActionRewardStateActionModel.Model

		local hiddenStateTensorArray =  NewRecurrentDeepDoubleStateActionRewardStateActionModel.hiddenStateTensorArray
		
		local WeightTensorArrayArray = NewRecurrentDeepDoubleStateActionRewardStateActionModel.WeightTensorArrayArray

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetTensor = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorTensor = NewRecurrentDeepDoubleStateActionRewardStateActionModel:generateLossTensor(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForUpdate], true)

		local selectedActionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForUpdate])

		Model:update(negatedTemporalDifferenceErrorTensor)

		WeightTensorArrayArray[selectedModelNumberForUpdate] = Model:getWeightTensorArray(true)

		Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForTargetTensor], true)

		local targetActionTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForTargetTensor])

		hiddenStateTensorArray[selectedModelNumberForUpdate] = selectedActionTensor

		hiddenStateTensorArray[selectedModelNumberForTargetTensor] = targetActionTensor

		return temporalDifferenceErrorTensor

	end)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setResetFunction(function()

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	return NewRecurrentDeepDoubleStateActionRewardStateActionModel

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:generateLossTensor(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

	local Model = self.Model

	local discountFactor = self.discountFactor
	
	local EligibilityTrace = self.EligibilityTrace
	
	local hiddenStateTensorArray = self.hiddenStateTensorArray

	local WeightTensorArrayArray = self.WeightTensorArrayArray

	local ClassesList = Model:getClassesList()

	local numberOfClasses = #ClassesList

	local outputDimensionSizeArray = {1, numberOfClasses}

	if (not hiddenStateTensorArray[1]) then hiddenStateTensorArray[1] = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

	if (not hiddenStateTensorArray[2]) then hiddenStateTensorArray[2] = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

	if (not WeightTensorArrayArray[1]) then WeightTensorArrayArray[1] = Model:getWeightTensorArray(true) end

	if (not WeightTensorArrayArray[2]) then WeightTensorArrayArray[2] = Model:getWeightTensorArray(true) end
	
	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForUpdate], true)

	local previousTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForUpdate])

	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForTargetTensor], true)
	
	local previousTargetQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForTargetTensor])

	local qTensor = Model:forwardPropagate(currentFeatureTensor, previousTargetQTensor)

	local discountedQTensor = AqwamTensorLibrary:multiply(discountFactor, qTensor, (1 - terminalStateValue))

	local targetTensor = AqwamTensorLibrary:add(rewardValue, discountedQTensor)

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:subtract(targetTensor, previousTensor)

	if (EligibilityTrace) then

		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		EligibilityTrace:increment(actionIndex, discountFactor, {1, #ClassesList})

		temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

	end

	return temporalDifferenceErrorTensor

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:setWeightTensorArray1(WeightTensorArray1, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.WeightTensorArrayArray[1] = WeightTensorArray1

	else

		self.WeightTensorArrayArray[1] = deepCopyTable(WeightTensorArray1)

	end

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:setWeightTensorArray2(WeightTensorArray2, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.WeightTensorArrayArray[2] = WeightTensorArray2

	else

		self.WeightTensorArrayArray[2] = deepCopyTable(WeightTensorArray2)

	end

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:getWeightTensorArray1(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.WeightTensorArrayArray[1]

	else

		return deepCopyTable(self.WeightTensorArrayArray[1])

	end

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:getWeightTensorArray2(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.WeightTensorArrayArray[2]

	else

		return deepCopyTable(self.WeightTensorArrayArray[2])

	end

end

return RecurrentDeepDoubleStateActionRewardStateActionModel
