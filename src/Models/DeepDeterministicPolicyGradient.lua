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

local DeepDeterministicPolicyGradientModel = {}

DeepDeterministicPolicyGradientModel.__index = DeepDeterministicPolicyGradientModel

setmetatable(DeepDeterministicPolicyGradientModel, ReinforcementLearningActorCriticBaseModel)

local defaultAveragingRate = 0.995

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function DeepDeterministicPolicyGradientModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDeepDeterministicPolicyGradientModel = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepDeterministicPolicyGradientModel, DeepDeterministicPolicyGradientModel)
	
	NewDeepDeterministicPolicyGradientModel:setName("DeepDeterministicPolicyGradient")
	
	NewDeepDeterministicPolicyGradientModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDeterministicPolicyGradientModel.TargetActorWeightTensorArray = parameterDictionary.TargetActorWeightTensorArray
	
	NewDeepDeterministicPolicyGradientModel.TargetCriticWeightTensorArray = parameterDictionary.TargetCriticWeightTensorArray
	
	NewDeepDeterministicPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)
		
		if (not previousActionNoiseTensor) then previousActionNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanTensor[1]}) end
		
		local ActorModel = NewDeepDeterministicPolicyGradientModel.ActorModel
		
		local CriticModel = NewDeepDeterministicPolicyGradientModel.CriticModel
		
		local averagingRate = NewDeepDeterministicPolicyGradientModel.averagingRate
		
		local TargetActorWeightTensorArray = NewDeepDeterministicPolicyGradientModel.TargetActorWeightTensorArray
		
		local TargetCriticWeightTensorArray = NewDeepDeterministicPolicyGradientModel.TargetCriticWeightTensorArray
		
		local PrimaryActorWeightTensorArray = ActorModel:getWeightTensorArray(true) or ActorModel:generateLayers()
		
		local PrimaryCriticWeightTensorArray = CriticModel:getWeightTensorArray(true) or CriticModel:generateLayers()
		
		TargetActorWeightTensorArray = TargetActorWeightTensorArray or PrimaryActorWeightTensorArray
		
		TargetCriticWeightTensorArray = TargetCriticWeightTensorArray or PrimaryCriticWeightTensorArray
		
		ActorModel:setWeightTensorArray(TargetActorWeightTensorArray)
		
		local targetCurrentActionMeanTensor = ActorModel:forwardPropagate(currentFeatureTensor)
		
		local targetCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(currentFeatureTensor, targetCurrentActionMeanTensor, 2)
		
		CriticModel:setWeightTensorArray(TargetCriticWeightTensorArray)
		
		local targetQValue = CriticModel:forwardPropagate(targetCriticActionMeanInputTensor)[1][1]
		
		local CriticWeightTensorArray = CriticModel:getWeightTensorArray(true)
	
		local yValue = rewardValue + (NewDeepDeterministicPolicyGradientModel.discountFactor * (1 - terminalStateValue) * targetQValue)
		
		ActorModel:setWeightTensorArray(PrimaryActorWeightTensorArray)

		ActorModel:forwardPropagate(previousFeatureTensor, true)
		
		local previousCriticActionInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, previousActionMeanTensor, 2)
		
		CriticModel:setWeightTensorArray(PrimaryCriticWeightTensorArray)
		
		local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputTensor, true)[1][1]

		local criticLoss = 2 * (currentQValue - yValue)
		
		local temporalDifferenceError = -criticLoss
		
		ActorModel:update(-currentQValue, true)
		
		CriticModel:update(temporalDifferenceError, true)
		
		PrimaryActorWeightTensorArray = ActorModel:getWeightTensorArray(true)
		
		PrimaryCriticWeightTensorArray = CriticModel:getWeightTensorArray(true)
		
		NewDeepDeterministicPolicyGradientModel.TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, PrimaryActorWeightTensorArray)
		
		NewDeepDeterministicPolicyGradientModel.TargetCriticWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArray, PrimaryCriticWeightTensorArray)

		return temporalDifferenceError
		
	end)
	
	NewDeepDeterministicPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewDeepDeterministicPolicyGradientModel:setResetFunction(function() end)
	
	return NewDeepDeterministicPolicyGradientModel
	
end

function DeepDeterministicPolicyGradientModel:setTargetActorWeightTensorArray(TargetActorWeightTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetActorWeightTensorArray = TargetActorWeightTensorArray

	else

		self.TargetActorWeightTensorArray = self:deepCopyTable(TargetActorWeightTensorArray)

	end

end

function DeepDeterministicPolicyGradientModel:setTargetCriticWeightTensorArray(TargetCriticWeightTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticWeightTensorArray = TargetCriticWeightTensorArray

	else

		self.TargetCriticWeightTensorArray = self:deepCopyTable(TargetCriticWeightTensorArray)

	end

end

function DeepDeterministicPolicyGradientModel:getTargetActorWeightTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetActorWeightTensorArray

	else

		return self:deepCopyTable(self.TargetActorWeightTensorArray)

	end

end

function DeepDeterministicPolicyGradientModel:getTargetCriticWeightTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticWeightTensorArray

	else

		return self:deepCopyTable(self.TargetCriticWeightTensorArray)

	end

end

return DeepDeterministicPolicyGradientModel
