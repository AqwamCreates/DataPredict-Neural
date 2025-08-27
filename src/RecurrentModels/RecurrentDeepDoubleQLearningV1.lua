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

RecurrentDeepDoubleQLearningModel = {}

RecurrentDeepDoubleQLearningModel.__index = RecurrentDeepDoubleQLearningModel

setmetatable(RecurrentDeepDoubleQLearningModel, DualRecurrentReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleQLearningModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleQLearningModel = DualRecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleQLearningModel, RecurrentDeepDoubleQLearningModel)

	NewRecurrentDeepDoubleQLearningModel:setName("RecurrentDeepDoubleQLearning")

	NewRecurrentDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentDeepDoubleQLearningModel.WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or {}

	NewRecurrentDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepDoubleQLearningModel.Model

		local hiddenStateTensorArray = NewRecurrentDeepDoubleQLearningModel.hiddenStateTensorArray
		
		local WeightTensorArrayArray = NewRecurrentDeepDoubleQLearningModel.WeightTensorArrayArray

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetTensor = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorTensor, temporalDifferenceError = NewRecurrentDeepDoubleQLearningModel:generateLossTensor(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.
		
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

	NewRecurrentDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)

		NewRecurrentDeepDoubleQLearningModel.EligibilityTrace:reset()

	end)

	NewRecurrentDeepDoubleQLearningModel:setResetFunction(function() 

		NewRecurrentDeepDoubleQLearningModel.EligibilityTrace:reset()

	end)

	return NewRecurrentDeepDoubleQLearningModel

end

function RecurrentDeepDoubleQLearningModel:generateLossTensor(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)

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

	local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForUpdate])

	Model:setWeightTensorArray(WeightTensorArrayArray[selectedModelNumberForTargetTensor], true)

	local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[selectedModelNumberForTargetTensor])

	local _, maxQValue = Model:predict(currentFeatureTensor, previousQTensor)

	local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

	local actionIndex = table.find(ClassesList, action)

	local lastValue = previousQTensor[1][actionIndex]

	local temporalDifferenceError = targetValue - lastValue

	local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

	temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError

	if (EligibilityTrace) then

		EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

		temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

	end

	return temporalDifferenceErrorTensor, temporalDifferenceError

end

return RecurrentDeepDoubleQLearningModel
