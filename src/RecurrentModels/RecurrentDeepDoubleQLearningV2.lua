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

local RecurrentDeepDoubleQLearningModel = {}

RecurrentDeepDoubleQLearningModel.__index = RecurrentDeepDoubleQLearningModel

setmetatable(RecurrentDeepDoubleQLearningModel, RecurrentDeepReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleQLearningModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleQLearningModel = RecurrentDeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleQLearningModel, RecurrentDeepDoubleQLearningModel)

	NewRecurrentDeepDoubleQLearningModel:setName("RecurrentDeepDoubleQLearningV2")

	NewRecurrentDeepDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewRecurrentDeepDoubleQLearningModel.primaryHiddenStateTensor = parameterDictionary.primaryHiddenStateTensor

	NewRecurrentDeepDoubleQLearningModel.targetHiddenStateTensor = parameterDictionary.targetHiddenStateTensor

	NewRecurrentDeepDoubleQLearningModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewRecurrentDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewRecurrentDeepDoubleQLearningModel.Model

		local discountFactor = NewRecurrentDeepDoubleQLearningModel.discountFactor

		local EligibilityTrace = NewRecurrentDeepDoubleQLearningModel.EligibilityTrace

		local TargetWeightTensorArray = NewRecurrentDeepDoubleQLearningModel.TargetWeightTensorArray
		
		local primaryHiddenStateTensor = NewRecurrentDeepDoubleQLearningModel.primaryHiddenStateTensor
		
		local targetHiddenStateTensor = NewRecurrentDeepDoubleQLearningModel.targetHiddenStateTensor

		local ClassesList = Model:getClassesList()
		
		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not primaryHiddenStateTensor) then primaryHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		if (not targetHiddenStateTensor) then targetHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		if (not PrimaryWeightTensorArray) then PrimaryWeightTensorArray = Model:generateLayers() end

		if (not TargetWeightTensorArray) then TargetWeightTensorArray = PrimaryWeightTensorArray end

		local primaryPreviousQTensor = Model:forwardPropagate(previousFeatureTensor, primaryHiddenStateTensor)

		local maximumPrimaryCurrentActionTensor = Model:predict(currentFeatureTensor, primaryPreviousQTensor)

		local primaryCurrentActionIndex = table.find(ClassesList, maximumPrimaryCurrentActionTensor[1][1])

		Model:setWeightTensorArray(TargetWeightTensorArray, true)
		
		local targetPreviousQTensor = Model:forwardPropagate(previousFeatureTensor, targetHiddenStateTensor)

		local targetCurrentQTensor = Model:forwardPropagate(currentFeatureTensor, targetPreviousQTensor)

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * targetCurrentQTensor[1][primaryCurrentActionIndex])

		local primaryPreviousActionIndex = table.find(ClassesList, previousAction)

		local primaryPreviousQValue = primaryPreviousQTensor[1][primaryPreviousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][primaryPreviousActionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(1, primaryPreviousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-RecurrentDeep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.

		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewRecurrentDeepDoubleQLearningModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDeepDoubleQLearningModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
		NewRecurrentDeepDoubleQLearningModel.primaryHiddenStateTensor = primaryPreviousQTensor

		NewRecurrentDeepDoubleQLearningModel.targetHiddenStateTensor = targetPreviousQTensor
		
		return temporalDifferenceErrorTensor

	end)

	NewRecurrentDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		local EligibilityTrace = NewRecurrentDeepDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepDoubleQLearningModel.primaryHiddenStateTensor = nil

		NewRecurrentDeepDoubleQLearningModel.targetHiddenStateTensor = nil

	end)

	NewRecurrentDeepDoubleQLearningModel:setResetFunction(function() 

		local EligibilityTrace = NewRecurrentDeepDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepDoubleQLearningModel.primaryHiddenStateTensor = nil

		NewRecurrentDeepDoubleQLearningModel.targetHiddenStateTensor = nil

	end)

	return NewRecurrentDeepDoubleQLearningModel

end

function RecurrentDeepDoubleQLearningModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:RecurrentDeepCopyTable(TargetWeightTensorArray)

	end

end

function RecurrentDeepDoubleQLearningModel:getTargetWeightTensorArray(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.TargetWeightTensorArray

	else

		return self:RecurrentDeepCopyTable(self.TargetWeightTensorArray)

	end

end

return RecurrentDeepDoubleQLearningModel
