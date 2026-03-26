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

local RecurrentDoubleStateActionRewardStateActionModel = {}

RecurrentDoubleStateActionRewardStateActionModel.__index = RecurrentDoubleStateActionRewardStateActionModel

setmetatable(RecurrentDoubleStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

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

function RecurrentDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDoubleStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDoubleStateActionRewardStateActionModel, RecurrentDoubleStateActionRewardStateActionModel)
	
	NewRecurrentDoubleStateActionRewardStateActionModel:setName("RecurrentDoubleStateActionRewardStateActionV2")

	NewRecurrentDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewRecurrentDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = parameterDictionary.primaryHiddenStateTensor

	NewRecurrentDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = parameterDictionary.targetHiddenStateTensor
	
	NewRecurrentDoubleStateActionRewardStateActionModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewRecurrentDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewRecurrentDoubleStateActionRewardStateActionModel.Model
		
		local discountFactor = NewRecurrentDoubleStateActionRewardStateActionModel.discountFactor

		local EligibilityTrace = NewRecurrentDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local TargetWeightTensorArray = NewRecurrentDoubleStateActionRewardStateActionModel.TargetWeightTensorArray
		
		local primaryHiddenStateTensor = NewRecurrentDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor
		
		local targetHiddenStateTensor = NewRecurrentDoubleStateActionRewardStateActionModel.targetHiddenStateTensor
		
		local ClassesList = Model:getClassesList()
		
		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not primaryHiddenStateTensor) then primaryHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		if (not targetHiddenStateTensor) then targetHiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end
		
		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		if (not PrimaryWeightTensorArray) then PrimaryWeightTensorArray = Model:generateLayers() end
		
		if (not TargetWeightTensorArray) then TargetWeightTensorArray = PrimaryWeightTensorArray end
		
		local primaryPreviousQTensor = Model:forwardPropagate(previousFeatureTensor, primaryHiddenStateTensor)
		
		Model:setWeightTensorArray(TargetWeightTensorArray, true)
		
		local targetPreviousQTensor = Model:forwardPropagate(previousFeatureTensor, targetHiddenStateTensor)
		
		local targetCurrentQTensor = Model:forwardPropagate(currentFeatureTensor, targetPreviousQTensor)

		local previousActionIndex = table.find(ClassesList, previousAction)

		local currentActionIndex = table.find(ClassesList, currentAction)

		local targetQValue = rewardValue + (discountFactor * targetCurrentQTensor[1][currentActionIndex] * (1 - terminalStateValue))
		
		local primaryPreviousQValue = primaryPreviousQTensor[1][previousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][previousActionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(1, previousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-Recurrent SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewRecurrentDoubleStateActionRewardStateActionModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDoubleStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
		NewRecurrentDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = primaryPreviousQTensor
		
		NewRecurrentDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = targetPreviousQTensor
		
		return temporalDifferenceErrorTensor

	end)
	
	NewRecurrentDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewRecurrentDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = nil

		NewRecurrentDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = nil
		
	end)

	NewRecurrentDoubleStateActionRewardStateActionModel:setResetFunction(function()
		
		local EligibilityTrace = NewRecurrentDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = nil
		
		NewRecurrentDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = nil
		
	end)

	return NewRecurrentDoubleStateActionRewardStateActionModel

end

function RecurrentDoubleStateActionRewardStateActionModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotRecurrentCopy)

	if (doNotRecurrentCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:RecurrentCopyTable(TargetWeightTensorArray)

	end

end

function RecurrentDoubleStateActionRewardStateActionModel:getTargetWeightTensorArray(doNotRecurrentCopy)

	if (doNotRecurrentCopy) then

		return self.TargetWeightTensorArray

	else

		return self:RecurrentCopyTable(self.TargetWeightTensorArray)

	end

end

return RecurrentDoubleStateActionRewardStateActionModel
