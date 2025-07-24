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

local DualRecurrentReinforcementLearningActorCriticBaseModel = require(script.Parent.DualRecurrentReinforcementLearningActorCriticBaseModel)

RecurrentTwinDelayedDeepDeterministicPolicyGradientModel = {}

RecurrentTwinDelayedDeepDeterministicPolicyGradientModel.__index = RecurrentTwinDelayedDeepDeterministicPolicyGradientModel

setmetatable(RecurrentTwinDelayedDeepDeterministicPolicyGradientModel, DualRecurrentReinforcementLearningActorCriticBaseModel)

local defaultAveragingRate = 0.995

local defaultNoiseClippingFactor = 0.5

local defaultPolicyDelayAmount = 3

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function RecurrentTwinDelayedDeepDeterministicPolicyGradientModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentTwinDelayedDeepDeterministicPolicyGradient = DualRecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentTwinDelayedDeepDeterministicPolicyGradient, RecurrentTwinDelayedDeepDeterministicPolicyGradientModel)

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient:setName("RecurrentTwinDelayedDeepDeterministicPolicyGradient")

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor = parameterDictionary.noiseClippingFactor or defaultNoiseClippingFactor

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount = parameterDictionary.policyDelayAmount or defaultPolicyDelayAmount

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.CriticWeightTensorArrayArray = parameterDictionary.CriticWeightTensorArrayArray or {}
	
	local targetCriticHiddenStateValueArray = {}

	local TargetCriticWeightTensorArrayArray = {}
	
	local previousPreviousActionArray = {}

	local currentNumberOfUpdate = 0
	
	local previousQValue

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensordimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensordimensionSizeArray) 

		end

		local criticHiddenStateValueArray = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.criticHiddenStateValueArray

		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

		criticHiddenStateValueArray[1] = criticHiddenStateValueArray[1] or 0

		criticHiddenStateValueArray[2] = criticHiddenStateValueArray[2] or 0
		
		targetCriticHiddenStateValueArray[1] = targetCriticHiddenStateValueArray[1] or 0
		
		targetCriticHiddenStateValueArray[2] = targetCriticHiddenStateValueArray[2] or 0

		local ActorModel = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.ActorModel

		local CriticModel = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.CriticModel

		local averagingRate = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.averagingRate

		local noiseClippingFactor = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor

		local CriticWeightTensorArrayArray = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.CriticWeightTensorArrayArray

		local noiseClipFunction = function(value) return math.clamp(value, -noiseClippingFactor, noiseClippingFactor) end
		
		local clippedPreviousActionNoiseTensor = AqwamTensorLibrary:applyFunction(noiseClipFunction, actionNoiseTensor)

		local currentActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanTensor[1]})

		local clippedCurrentActionNoiseTensor = AqwamTensorLibrary:applyFunction(noiseClipFunction, currentActionNoiseTensor)

		local previousActionTensor = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

		previousActionTensor = AqwamTensorLibrary:add(previousActionTensor, actionMeanTensor)

		local previousActionArray = previousActionTensor[1] 
		
		local previousLowestActionValue = -math.huge
		
		local previousHighestActionValue = math.huge
		
		if (previousPreviousActionArray) then
			
			previousLowestActionValue = math.min(table.unpack(previousPreviousActionArray))
			
			previousHighestActionValue = math.max(table.unpack(previousPreviousActionArray))
			
		end

		local lowestActionValue = math.min(table.unpack(previousActionArray))

		local highestActionValue = math.max(table.unpack(previousActionArray))

		local currentActionMeanTensor = ActorModel:forwardPropagate(currentFeatureTensor, actionMeanTensor)

		local ActorWeightTensorArray = ActorModel:getWeightTensorArray(true)
		
		local previousTargetActionTensorPart1 = AqwamTensorLibrary:add(actionMeanTensor, clippedPreviousActionNoiseTensor)

		local targetActionTensorPart1 = AqwamTensorLibrary:add(currentActionMeanTensor, clippedCurrentActionNoiseTensor)
		
		local previousActionClipFunction = function(value)
			
			if (previousLowestActionValue ~= previousLowestActionValue) or (previousHighestActionValue ~= previousHighestActionValue) then

				error("Received Nan values.")

			elseif (previousLowestActionValue < previousHighestActionValue) then

				return math.clamp(value, previousLowestActionValue, previousHighestActionValue) 

			elseif (previousLowestActionValue > previousHighestActionValue) then

				return math.clamp(value, previousHighestActionValue, previousLowestActionValue)

			else

				return previousLowestActionValue

			end
			
		end

		local actionClipFunction = function(value)

			if (lowestActionValue ~= lowestActionValue) or (highestActionValue ~= highestActionValue) then

				error("Received Nan values.")

			elseif (lowestActionValue < highestActionValue) then

				return math.clamp(value, lowestActionValue, highestActionValue) 

			elseif (lowestActionValue > highestActionValue) then

				return math.clamp(value, highestActionValue, lowestActionValue)

			else

				return lowestActionValue

			end

		end
		
		local previousTargetActionTensor = AqwamTensorLibrary:applyFunction(previousActionClipFunction, previousTargetActionTensorPart1)

		local targetActionTensor = AqwamTensorLibrary:applyFunction(actionClipFunction, targetActionTensorPart1)
		
		local previousCriticActionInputTensor = AqwamTensorLibrary:concatenate(previousActionTensor, previousTargetActionTensor, 2)

		local targetCriticActionInputTensor = AqwamTensorLibrary:concatenate(currentFeatureTensor, targetActionTensor, 2)

		local currentCriticValueArray = {}

		for i = 1, 2, 1 do 

			CriticModel:setWeightTensorArray(TargetCriticWeightTensorArrayArray[i])
			
			local previousCriticTensor = CriticModel:forwardPropagate(previousCriticActionInputTensor, {{targetCriticHiddenStateValueArray[i]}})

			currentCriticValueArray[i] = CriticModel:forwardPropagate(targetCriticActionInputTensor, previousCriticTensor)[1][1] 

			local CriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

			TargetCriticWeightTensorArrayArray[i] = CriticWeightTensorArray
			
			targetCriticHiddenStateValueArray[i] = previousCriticTensor[1][1]

		end

		local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))

		local yValuePart1 = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.discountFactor * (1 - terminalStateValue) * minimumCurrentCriticValue

		local yValue = rewardValue + yValuePart1

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2}, 0)

		local previousCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, actionMeanTensor, 2)

		for i = 1, 2, 1 do 

			CriticModel:setWeightTensorArray(CriticWeightTensorArrayArray[i], true)

			local previousCriticValue = CriticModel:forwardPropagate(previousCriticActionMeanInputTensor, {{criticHiddenStateValueArray[i]}})[1][1] 

			local criticLoss = previousCriticValue - yValue

			temporalDifferenceErrorTensor[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

			CriticModel:update(criticLoss)

			CriticWeightTensorArrayArray[i] = CriticModel:getWeightTensorArray(true)
			
			criticHiddenStateValueArray[i] = previousCriticValue

		end

		currentNumberOfUpdate = currentNumberOfUpdate + 1

		if ((currentNumberOfUpdate % NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount) == 0) then
			
			local actorHiddenStateTensorArray = NewRecurrentTwinDelayedDeepDeterministicPolicyGradient.actorHiddenStateTensorArray
			
			actorHiddenStateTensorArray[1] = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

			local actionTensor = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

			actionTensor = AqwamTensorLibrary:add(actionTensor, actionMeanTensor)

			local previousCriticActionInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, actionTensor, 2)
			
			previousQValue = previousQValue or 0

			CriticModel:setWeightTensorArray(CriticWeightTensorArrayArray[1])

			local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputTensor, {{previousQValue}})[1][1]

			local previousActionTensor = ActorModel:forwardPropagate(previousFeatureTensor, actorHiddenStateTensorArray[1])

			ActorModel:update(-currentQValue)

			for i = 1, 2, 1 do TargetCriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArrayArray[i], CriticWeightTensorArrayArray[i]) end

			local TargetActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

			TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, ActorWeightTensorArray)

			ActorModel:setWeightTensorArray(TargetActorWeightTensorArray, true)
			
			actorHiddenStateTensorArray[1] = previousActionTensor
			
			previousQValue = currentQValue

		end

		return temporalDifferenceErrorTensor

	end)

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient:setEpisodeUpdateFunction(function()
		
		targetCriticHiddenStateValueArray = {}

		currentNumberOfUpdate = 0
		
		previousQValue = 0

	end)

	NewRecurrentTwinDelayedDeepDeterministicPolicyGradient:setResetFunction(function() 
		
		targetCriticHiddenStateValueArray = {}

		currentNumberOfUpdate = 0
		
		previousQValue = 0

	end)

	return NewRecurrentTwinDelayedDeepDeterministicPolicyGradient

end

function RecurrentTwinDelayedDeepDeterministicPolicyGradientModel:setCrtiticWeightTensorArray1(CriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticWeightTensorArrayArray[1] = CriticWeightTensorArray1

	else

		self.CriticWeightTensorArrayArray[1] = self:deepCopyTable(CriticWeightTensorArray1)

	end

end

function RecurrentTwinDelayedDeepDeterministicPolicyGradientModel:setCriticWeightTensorArray2(CriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticWeightTensorArrayArray[2] = CriticWeightTensorArray2

	else

		self.CriticWeightTensorArrayArray[2] = self:deepCopyTable(CriticWeightTensorArray2)

	end

end

function RecurrentTwinDelayedDeepDeterministicPolicyGradientModel:getCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.CriticWeightTensorArrayArray[1])

	end

end

function RecurrentTwinDelayedDeepDeterministicPolicyGradientModel:getCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.CriticWeightTensorArrayArray[2])

	end

end

return RecurrentTwinDelayedDeepDeterministicPolicyGradientModel