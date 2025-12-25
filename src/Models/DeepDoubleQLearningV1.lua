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

local DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, ReinforcementLearningBaseModel)

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

function DeepDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearning")
	
	NewDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepDoubleQLearningModel.WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or {}

	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewDeepDoubleQLearningModel.Model
		
		local WeightTensorArrayArray = NewDeepDoubleQLearningModel.WeightTensorArrayArray

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetTensor = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorTensor, temporalDifferenceError = NewDeepDoubleQLearningModel:generateLossTensor(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.
		
		Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForUpdate], true)
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		WeightTensorArrayArray[selectedModelNumberForUpdate] = Model:getWeightTensorArray(true)

		return temporalDifferenceErrorTensor

	end)

	NewDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewDeepDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepDoubleQLearningModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:generateLossTensor(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local EligibilityTrace = self.EligibilityTrace
	
	local WeightTensorArrayArray = self.WeightTensorArrayArray

	if (not WeightTensorArrayArray[1]) then WeightTensorArrayArray[1] = Model:getWeightTensorArray(true) end

	if (not WeightTensorArrayArray[2]) then WeightTensorArrayArray[2] = Model:getWeightTensorArray(true) end
	
	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForUpdate], true)

	local previousQTensor = Model:forwardPropagate(previousFeatureTensor)

	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForTargetTensor], true)

	local _, maxQValue = Model:predict(currentFeatureTensor)

	local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

	local ClassesList = Model:getClassesList()

	local numberOfClasses = #ClassesList

	local actionIndex = table.find(ClassesList, previousAction)

	local lastValue = previousQTensor[1][actionIndex]

	local temporalDifferenceError = targetValue - lastValue
	
	local outputDimensionSizeArray = {1, numberOfClasses}

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

	temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
	
	if (EligibilityTrace) then

		EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

		temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

	end

	return temporalDifferenceErrorTensor, temporalDifferenceError

end

function DeepDoubleQLearningModel:setWeightTensorArray1(WeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[1] = WeightTensorArray1

	else

		self.WeightTensorArrayArray[1] = deepCopyTable(WeightTensorArray1)

	end

end

function DeepDoubleQLearningModel:setWeightTensorArray2(WeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[2] = WeightTensorArray2

	else

		self.WeightTensorArrayArray[2] = deepCopyTable(WeightTensorArray2)

	end

end

function DeepDoubleQLearningModel:getWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[1]

	else

		return deepCopyTable(self.WeightTensorArrayArray[1])

	end

end

function DeepDoubleQLearningModel:getWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[2]

	else

		return deepCopyTable(self.WeightTensorArrayArray[2])

	end

end

return DeepDoubleQLearningModel
