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

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

local TwinDelayedDeepDeterministicPolicyGradientModel = {}

TwinDelayedDeepDeterministicPolicyGradientModel.__index = TwinDelayedDeepDeterministicPolicyGradientModel

setmetatable(TwinDelayedDeepDeterministicPolicyGradientModel, ReinforcementLearningActorCriticBaseModel)

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

function TwinDelayedDeepDeterministicPolicyGradientModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTwinDelayedDeepDeterministicPolicyGradient = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewTwinDelayedDeepDeterministicPolicyGradient, TwinDelayedDeepDeterministicPolicyGradientModel)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setName("TwinDelayedDeepDeterministicPolicyGradient")
	
	NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor = parameterDictionary.noiseClippingFactor or defaultNoiseClippingFactor

	NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount = parameterDictionary.policyDelayAmount or defaultPolicyDelayAmount
	
	NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTwinDelayedDeepDeterministicPolicyGradient.TargetActorWeightTensorArray = parameterDictionary.TargetActorWeightTensorArray
	
	NewTwinDelayedDeepDeterministicPolicyGradient.PrimaryCriticWeightTensorArrayArray = parameterDictionary.PrimaryCriticWeightTensorArrayArray or {}
	
	NewTwinDelayedDeepDeterministicPolicyGradient.TargetCriticWeightTensorArrayArray = parameterDictionary.TargetCriticWeightTensorArrayArray or {}
	
	local currentNumberOfUpdates = 0
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)
		
		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end
		
		local ActorModel = NewTwinDelayedDeepDeterministicPolicyGradient.ActorModel
		
		local CriticModel = NewTwinDelayedDeepDeterministicPolicyGradient.CriticModel
		
		local averagingRate = NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate
		
		local noiseClippingFactor = NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor
		
		local TargetActorWeightTensorArray = NewTwinDelayedDeepDeterministicPolicyGradient.TargetActorWeightTensorArray
		
		local PrimaryCriticWeightTensorArrayArray = NewTwinDelayedDeepDeterministicPolicyGradient.PrimaryCriticWeightTensorArrayArray
		
		local TargetCriticWeightTensorArrayArray = NewTwinDelayedDeepDeterministicPolicyGradient.TargetCriticWeightTensorArrayArray
		
		local PrimaryActorWeightTensorArray = ActorModel:getWeightTensorArray(true) or ActorModel:generateLayers()
		
		PrimaryCriticWeightTensorArrayArray[1] = CriticModel:getWeightTensorArray(true) or CriticModel:generateLayers()
		
		PrimaryCriticWeightTensorArrayArray[2] = CriticModel:getWeightTensorArray(true) or CriticModel:generateLayers()
		
		TargetActorWeightTensorArray = TargetActorWeightTensorArray or PrimaryActorWeightTensorArray
		
		TargetCriticWeightTensorArrayArray[1] = TargetCriticWeightTensorArrayArray[1] or PrimaryCriticWeightTensorArrayArray[1]
		
		TargetCriticWeightTensorArrayArray[2] = TargetCriticWeightTensorArrayArray[2] or PrimaryCriticWeightTensorArrayArray[2]
		
		local noiseClipFunction = function(value) return math.clamp(value, -noiseClippingFactor, noiseClippingFactor) end
		
		local currentActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]})
		
		local clippedCurrentActionNoiseTensor = AqwamTensorLibrary:applyFunction(noiseClipFunction, currentActionNoiseTensor)
		
		local previousActionTensor = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		previousActionTensor = AqwamTensorLibrary:add(previousActionTensor, previousActionMeanTensor)

		local unwrappedPreviousActionTensor = previousActionTensor[1]

		local lowestActionValue = math.min(table.unpack(unwrappedPreviousActionTensor))

		local highestActionValue = math.max(table.unpack(unwrappedPreviousActionTensor))
		
		local ActorWeightTensorArray = ActorModel:getWeightTensorArray(true)
		
		ActorModel:setWeightTensorArray(TargetActorWeightTensorArray)
		
		local targetCurrentActionMeanTensor = ActorModel:forwardPropagate(currentFeatureTensor)
		
		local targetActionTensorPart1 = AqwamTensorLibrary:add(targetCurrentActionMeanTensor, clippedCurrentActionNoiseTensor)
		
		local actionClipFunction = function(value)
			
			if (lowestActionValue ~= lowestActionValue) or (highestActionValue ~= highestActionValue) then
				
				error("Received nan values.")
			
			elseif (lowestActionValue < highestActionValue) then
				
				return math.clamp(value, lowestActionValue, highestActionValue) 
				
			elseif (lowestActionValue > highestActionValue) then
				
				return math.clamp(value, highestActionValue, lowestActionValue)
				
			else
				
				return lowestActionValue
				
			end
			
		end
		
		local targetActionTensor = AqwamTensorLibrary:applyFunction(actionClipFunction, targetActionTensorPart1)
		
		local targetCriticActionInputTensor = AqwamTensorLibrary:concatenate(currentFeatureTensor, targetActionTensor, 2)
		
		local currentCriticValueArray = {}
		
		for i = 1, 2, 1 do 

			CriticModel:setWeightTensorArray(TargetCriticWeightTensorArrayArray[i])

			currentCriticValueArray[i] = CriticModel:forwardPropagate(targetCriticActionInputTensor)[1][1] 

		end

		local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))
		
		local yValuePart1 = NewTwinDelayedDeepDeterministicPolicyGradient.discountFactor * (1 - terminalStateValue) * minimumCurrentCriticValue
		
		local yValue = rewardValue + yValuePart1
		
		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2}, 0)

		local previousCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, previousActionMeanTensor, 2)
		
		for i = 1, 2, 1 do

			CriticModel:setWeightTensorArray(PrimaryCriticWeightTensorArrayArray[i], true)

			local previousCriticValue = CriticModel:forwardPropagate(previousCriticActionMeanInputTensor, true)[1][1] 

			local criticLoss = 2 * (previousCriticValue - yValue)

			temporalDifferenceErrorTensor[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

			CriticModel:update(criticLoss, true)
			
			PrimaryCriticWeightTensorArrayArray[i] = CriticModel:getWeightTensorArray(true)

		end
		
		currentNumberOfUpdates = currentNumberOfUpdates + 1
		
		if ((currentNumberOfUpdates % NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount) == 0) then
			
			CriticModel:setWeightTensorArray(PrimaryCriticWeightTensorArrayArray[1], true)

			local currentQValue = CriticModel:forwardPropagate(previousCriticActionMeanInputTensor, true)[1][1]
			
			ActorModel:setWeightTensorArray(PrimaryActorWeightTensorArray, true)

			ActorModel:forwardPropagate(previousFeatureTensor, true)

			ActorModel:update(-currentQValue, true)

			for i = 1, 2, 1 do TargetCriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArrayArray[i], PrimaryCriticWeightTensorArrayArray[i]) end
			
			local PrimaryActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

			NewTwinDelayedDeepDeterministicPolicyGradient.TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, PrimaryActorWeightTensorArray)
			
		end

		return temporalDifferenceErrorTensor
		
	end)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setEpisodeUpdateFunction(function() 
		
		currentNumberOfUpdates = 0
		
	end)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setResetFunction(function() 
		
		currentNumberOfUpdates = 0
		
	end)
	
	return NewTwinDelayedDeepDeterministicPolicyGradient
	
end

function TwinDelayedDeepDeterministicPolicyGradientModel:setTargetActorWeightTensorArray(TargetActorWeightTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetActorWeightTensorArray = TargetActorWeightTensorArray

	else

		self.TargetActorWeightTensorArray = self:deepCopyTable(TargetActorWeightTensorArray)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setPrimaryCrtiticWeightTensorArray1(PrimaryCriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticWeightTensorArrayArray[1] = PrimaryCriticWeightTensorArray1

	else

		self.PrimaryCriticWeightTensorArrayArray[1] = self:deepCopyTable(PrimaryCriticWeightTensorArray1)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setPrimaryCriticWeightTensorArray2(PrimaryCriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticWeightTensorArrayArray[2] = PrimaryCriticWeightTensorArray2

	else

		self.PrimaryCriticWeightTensorArrayArray[2] = self:deepCopyTable(PrimaryCriticWeightTensorArray2)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setTargetCrtiticWeightTensorArray1(TargetCriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticWeightTensorArrayArray[1] = TargetCriticWeightTensorArray1

	else

		self.TargetCriticWeightTensorArrayArray[1] = self:deepCopyTable(TargetCriticWeightTensorArray1)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setTargetCriticWeightTensorArray2(TargetCriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticWeightTensorArrayArray[2] = TargetCriticWeightTensorArray2

	else

		self.TargetCriticWeightTensorArrayArray[2] = self:deepCopyTable(TargetCriticWeightTensorArray2)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getTargetActorWeightTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetActorWeightTensorArray

	else

		return self:deepCopyTable(self.TargetActorWeightTensorArray)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getPrimaryCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.PrimaryCriticWeightTensorArrayArray[1])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getPrimaryCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.PrimaryCriticWeightTensorArrayArray[2])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getTargetCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.TargetCriticWeightTensorArrayArray[1])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getTargetCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.TargetCriticWeightTensorArrayArray[2])

	end

end

return TwinDelayedDeepDeterministicPolicyGradientModel
