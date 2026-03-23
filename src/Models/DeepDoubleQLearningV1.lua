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

local defaultAveragingRate = 0.01

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, PrimaryWeightTensorArray[layer])

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, TargetWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(PrimaryWeightTensorArrayPart, TargetWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function DeepDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearningV2")
	
	NewDeepDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepDoubleQLearningModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local discountFactor = NewDeepDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepDoubleQLearningModel.EligibilityTrace
		
		local TargetWeightTensorArray = NewDeepDoubleQLearningModel.TargetWeightTensorArray
		
		local ClassesList = Model:getClassesList()
		
		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		if (not PrimaryWeightTensorArray) then PrimaryWeightTensorArray = Model:generateLayers() end
		
		if (not TargetWeightTensorArray) then TargetWeightTensorArray = PrimaryWeightTensorArray end
		
		local primaryPreviousQTensor = Model:forwardPropagate(previousFeatureTensor)
		
		local maximumPrimaryCurrentActionTensor = Model:predict(currentFeatureTensor)
		
		local primaryCurrentActionIndex = table.find(ClassesList, maximumPrimaryCurrentActionTensor[1][1])
		
		Model:setWeightTensorArray(TargetWeightTensorArray, true)
		
		local targetCurrentQTensor = Model:forwardPropagate(currentFeatureTensor)

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * targetCurrentQTensor[1][primaryCurrentActionIndex])

		local primaryPreviousActionIndex = table.find(ClassesList, previousAction)

		local primaryPreviousQValue = primaryPreviousQTensor[1][primaryPreviousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][primaryPreviousActionIndex] = temporalDifferenceError
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(1, primaryPreviousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewDeepDoubleQLearningModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewDeepDoubleQLearningModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
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

function DeepDoubleQLearningModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:deepCopyTable(TargetWeightTensorArray)

	end
	
end

function DeepDoubleQLearningModel:getTargetWeightTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetWeightTensorArray

	else

		return self:deepCopyTable(self.TargetWeightTensorArray)

	end

end

return DeepDoubleQLearningModel
