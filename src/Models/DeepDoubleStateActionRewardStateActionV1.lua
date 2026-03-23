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

function DeepDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearningV1")
	
	NewDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepDoubleQLearningModel.WeightTensorArrayArray = parameterDictionary.WeightTensorArrayArray or {}
	
	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetTensor = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorTensor, temporalDifferenceError = NewDeepDoubleQLearningModel:generateTemporalDifferenceErrorTensor(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		NewDeepDoubleQLearningModel:loadWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForUpdate)
		
		Model:forwardPropagate(previousFeatureTensor, true)
		
		Model:update(negatedTemporalDifferenceErrorTensor, true)

		NewDeepDoubleQLearningModel:saveWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError
		
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

function DeepDoubleQLearningModel:saveWeightTensorArrayFromWeightTensorArrayArray(index)

	self.WeightTensorArrayArray[index] = self.Model:getWeightTensorArray()

end

function DeepDoubleQLearningModel:loadWeightTensorArrayFromWeightTensorArrayArray(index)

	self.Model:setWeightTensorArray(self.WeightTensorArrayArray[index], true)

end

function DeepDoubleQLearningModel:generateTemporalDifferenceErrorTensor(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, terminalStateValue, selectedModelNumberForTargetTensor, selectedModelNumberForUpdate)
	
	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local EligibilityTrace = self.EligibilityTrace
	
	self:loadWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForUpdate)
	
	local previousQTensor = Model:forwardPropagate(previousFeatureTensor)
	
	self:loadWeightTensorArrayFromWeightTensorArrayArray(selectedModelNumberForTargetTensor)

	local _, currentMaximumQValue = Model:predict(currentFeatureTensor)

	local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * currentMaximumQValue[1][1])
	
	local ClassesList = Model:getClassesList()
	
	local numberOfClasses = #ClassesList

	local actionIndex = table.find(ClassesList, previousAction)
	
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

function DeepDoubleQLearningModel:setWeightTensorArray1(WeightTensorArray1, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.WeightTensorArrayArray[1] = WeightTensorArray1
		
	else
		
		self.WeightTensorArrayArray[1] = self:deepCopyTable(WeightTensorArray1)
		
	end

end

function DeepDoubleQLearningModel:setWeightTensorArray2(WeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[2] = WeightTensorArray2

	else

		self.WeightTensorArrayArray[2] = self:deepCopyTable(WeightTensorArray2)

	end

end

function DeepDoubleQLearningModel:getWeightTensorArray1(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.WeightTensorArrayArray[1]
		
	else
		
		return self:deepCopyTable(self.WeightTensorArrayArray[1])
		
	end

end

function DeepDoubleQLearningModel:getWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.WeightTensorArrayArray[2])

	end

end

return DeepDoubleQLearningModel
