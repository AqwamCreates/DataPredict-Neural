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

TwinDelayedDeepDeterministicPolicyGradientModel = {}

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

	NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor = parameterDictionary.noiseClippingFactor or defaultNoiseClippingFactor

	NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount = parameterDictionary.policyDelayAmount or defaultPolicyDelayAmount

	NewTwinDelayedDeepDeterministicPolicyGradient.CriticWeightTensorArrayArray = parameterDictionary.CriticWeightTensorArrayArray or {}

	local TargetCriticWeightTensorArrayArray = {}

	local currentNumberOfUpdate = 0

	NewTwinDelayedDeepDeterministicPolicyGradient:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, actionMeanTensor, actionStandardDeviationTensor, actionNoiseTensor, rewardValue, currentFeatureTensor, terminalStateValue)

		if (not actionNoiseTensor) then

			local actionTensordimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanTensor)

			actionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensordimensionSizeArray) 

		end

		local ActorModel = NewTwinDelayedDeepDeterministicPolicyGradient.ActorModel

		local CriticModel = NewTwinDelayedDeepDeterministicPolicyGradient.CriticModel

		local averagingRate = NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate

		local noiseClippingFactor = NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor

		local CriticWeightTensorArrayArray = NewTwinDelayedDeepDeterministicPolicyGradient.CriticWeightTensorArrayArray

		local noiseClipFunction = function(value) return math.clamp(value, -noiseClippingFactor, noiseClippingFactor) end

		local currentActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanTensor[1]})

		local clippedCurrentActionNoiseTensor = AqwamTensorLibrary:applyFunction(noiseClipFunction, currentActionNoiseTensor)

		local previousActionTensor = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

		previousActionTensor = AqwamTensorLibrary:add(previousActionTensor, actionMeanTensor)

		local previousActionArray = previousActionTensor[1] 

		local lowestActionValue = math.min(table.unpack(previousActionArray))

		local highestActionValue = math.max(table.unpack(previousActionArray))

		local currentActionMeanTensor = ActorModel:forwardPropagate(currentFeatureTensor, true)

		local ActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local targetActionTensorPart1 = AqwamTensorLibrary:add(currentActionMeanTensor, clippedCurrentActionNoiseTensor)

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

			local CriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

			TargetCriticWeightTensorArrayArray[i] = CriticWeightTensorArray

		end

		local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))

		local yValuePart1 = NewTwinDelayedDeepDeterministicPolicyGradient.discountFactor * (1 - terminalStateValue) * minimumCurrentCriticValue

		local yValue = rewardValue + yValuePart1

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2}, 0)

		local previousCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, actionMeanTensor, 2)

		for i = 1, 2, 1 do 

			CriticModel:setWeightTensorArray(CriticWeightTensorArrayArray[i], true)

			local previousCriticValue = CriticModel:forwardPropagate(previousCriticActionMeanInputTensor, true)[1][1] 

			local criticLoss = previousCriticValue - yValue

			temporalDifferenceErrorTensor[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

			CriticModel:update(criticLoss, true)

			CriticWeightTensorArrayArray[i] = CriticModel:getWeightTensorArray(true)

		end

		currentNumberOfUpdate = currentNumberOfUpdate + 1

		if ((currentNumberOfUpdate % NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount) == 0) then

			local actionTensor = AqwamTensorLibrary:multiply(actionStandardDeviationTensor, actionNoiseTensor)

			actionTensor = AqwamTensorLibrary:add(actionTensor, actionMeanTensor)

			local previousCriticActionInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, actionTensor, 2)

			CriticModel:setWeightTensorArray(CriticWeightTensorArrayArray[1], true)

			local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputTensor, true)[1][1]

			ActorModel:forwardPropagate(previousFeatureTensor, true)

			ActorModel:update(-currentQValue, true)

			for i = 1, 2, 1 do TargetCriticWeightTensorArrayArray[i] = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArrayArray[i], CriticWeightTensorArrayArray[i]) end

			local TargetActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

			TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, ActorWeightTensorArray)

			ActorModel:setWeightTensorArray(TargetActorWeightTensorArray, true)

		end

		return temporalDifferenceErrorTensor

	end)

	NewTwinDelayedDeepDeterministicPolicyGradient:setEpisodeUpdateFunction(function() 

		currentNumberOfUpdate = 0

	end)

	NewTwinDelayedDeepDeterministicPolicyGradient:setResetFunction(function() 

		currentNumberOfUpdate = 0

	end)

	return NewTwinDelayedDeepDeterministicPolicyGradient

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setCrtiticWeightTensorArray1(CriticWeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticWeightTensorArrayArray[1] = CriticWeightTensorArray1

	else

		self.CriticWeightTensorArrayArray[1] = self:deepCopyTable(CriticWeightTensorArray1)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setCriticWeightTensorArray2(CriticWeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticWeightTensorArrayArray[2] = CriticWeightTensorArray2

	else

		self.CriticWeightTensorArrayArray[2] = self:deepCopyTable(CriticWeightTensorArray2)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getCriticWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticWeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.CriticWeightTensorArrayArray[1])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getCriticWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticWeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.CriticWeightTensorArrayArray[2])

	end

end

return TwinDelayedDeepDeterministicPolicyGradientModel