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

local RecurrentDoubleQLearningModel = {}

RecurrentDoubleQLearningModel.__index = RecurrentDoubleQLearningModel

setmetatable(RecurrentDoubleQLearningModel, RecurrentReinforcementLearningBaseModel)

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

function RecurrentDoubleQLearningModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDoubleQLearningModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDoubleQLearningModel, RecurrentDoubleQLearningModel)

	NewRecurrentDoubleQLearningModel:setName("RecurrentDoubleQLearningV2")

	NewRecurrentDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewRecurrentDoubleQLearningModel.primaryHiddenStateTensor = parameterDictionary.primaryHiddenStateTensor

	NewRecurrentDoubleQLearningModel.targetHiddenStateTensor = parameterDictionary.targetHiddenStateTensor

	NewRecurrentDoubleQLearningModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewRecurrentDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewRecurrentDoubleQLearningModel.Model

		local discountFactor = NewRecurrentDoubleQLearningModel.discountFactor

		local EligibilityTrace = NewRecurrentDoubleQLearningModel.EligibilityTrace

		local TargetWeightTensorArray = NewRecurrentDoubleQLearningModel.TargetWeightTensorArray
		
		local primaryHiddenStateTensor = NewRecurrentDoubleQLearningModel.primaryHiddenStateTensor
		
		local targetHiddenStateTensor = NewRecurrentDoubleQLearningModel.targetHiddenStateTensor

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

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-Recurrent Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.

		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)

		Model:forwardPropagate(previousFeatureTensor, primaryHiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewRecurrentDoubleQLearningModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDoubleQLearningModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
		NewRecurrentDoubleQLearningModel.primaryHiddenStateTensor = primaryPreviousQTensor

		NewRecurrentDoubleQLearningModel.targetHiddenStateTensor = targetPreviousQTensor
		
		return temporalDifferenceErrorTensor

	end)

	NewRecurrentDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local EligibilityTrace = NewRecurrentDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDoubleQLearningModel.primaryHiddenStateTensor = nil

		NewRecurrentDoubleQLearningModel.targetHiddenStateTensor = nil

	end)

	NewRecurrentDoubleQLearningModel:setResetFunction(function()

		local EligibilityTrace = NewRecurrentDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDoubleQLearningModel.primaryHiddenStateTensor = nil

		NewRecurrentDoubleQLearningModel.targetHiddenStateTensor = nil

	end)

	return NewRecurrentDoubleQLearningModel

end

function RecurrentDoubleQLearningModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotRecurrentCopy)

	if (doNotRecurrentCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:RecurrentCopyTable(TargetWeightTensorArray)

	end

end

function RecurrentDoubleQLearningModel:getTargetWeightTensorArray(doNotRecurrentCopy)

	if (doNotRecurrentCopy) then

		return self.TargetWeightTensorArray

	else

		return self:RecurrentCopyTable(self.TargetWeightTensorArray)

	end

end

return RecurrentDoubleQLearningModel
