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

local RecurrentDeepDoubleStateActionRewardStateActionModel = {}

RecurrentDeepDoubleStateActionRewardStateActionModel.__index = RecurrentDeepDoubleStateActionRewardStateActionModel

setmetatable(RecurrentDeepDoubleStateActionRewardStateActionModel, RecurrentDeepReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleStateActionRewardStateActionModel = RecurrentDeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleStateActionRewardStateActionModel, RecurrentDeepDoubleStateActionRewardStateActionModel)
	
	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setName("RecurrentDeepDoubleStateActionRewardStateActionV2")

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewRecurrentDeepDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = parameterDictionary.primaryHiddenStateTensor

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = parameterDictionary.targetHiddenStateTensor
	
	NewRecurrentDeepDoubleStateActionRewardStateActionModel.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewRecurrentDeepDoubleStateActionRewardStateActionModel.Model
		
		local discountFactor = NewRecurrentDeepDoubleStateActionRewardStateActionModel.discountFactor

		local EligibilityTrace = NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local TargetWeightTensorArray = NewRecurrentDeepDoubleStateActionRewardStateActionModel.TargetWeightTensorArray
		
		local primaryHiddenStateTensor = NewRecurrentDeepDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor
		
		local targetHiddenStateTensor = NewRecurrentDeepDoubleStateActionRewardStateActionModel.targetHiddenStateTensor
		
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
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-RecurrentDeep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:setWeightTensorArray(PrimaryWeightTensorArray, true)
		
		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)
		
		NewRecurrentDeepDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = primaryPreviousQTensor
		
		NewRecurrentDeepDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = targetPreviousQTensor
		
		return temporalDifferenceErrorTensor

	end)
	
	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = nil

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = nil
		
	end)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setResetFunction(function()
		
		local EligibilityTrace = NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepDoubleStateActionRewardStateActionModel.primaryHiddenStateTensor = nil
		
		NewRecurrentDeepDoubleStateActionRewardStateActionModel.targetHiddenStateTensor = nil
		
	end)

	return NewRecurrentDeepDoubleStateActionRewardStateActionModel

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:setTargetWeightTensorArray(TargetWeightTensorArray, doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:RecurrentDeepCopyTable(TargetWeightTensorArray)

	end

end

function RecurrentDeepDoubleStateActionRewardStateActionModel:getTargetWeightTensorArray(doNotRecurrentDeepCopy)

	if (doNotRecurrentDeepCopy) then

		return self.TargetWeightTensorArray

	else

		return self:RecurrentDeepCopyTable(self.TargetWeightTensorArray)

	end

end

return RecurrentDeepDoubleStateActionRewardStateActionModel
