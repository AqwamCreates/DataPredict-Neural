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

	NewDeepDeterministicPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then

			local actionTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensorDimensionSizeArray) 

		end

		local ActorModel = NewDeepDeterministicPolicyGradientModel.ActorModel

		local CriticModel = NewDeepDeterministicPolicyGradientModel.CriticModel

		local averagingRate = NewDeepDeterministicPolicyGradientModel.averagingRate

		local currentActionMeanTensor = ActorModel:forwardPropagate(currentFeatureTensor, true)

		local ActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local targetCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(currentFeatureTensor, currentActionMeanTensor, 2)

		local targetQValue = CriticModel:forwardPropagate(targetCriticActionMeanInputTensor, true)[1][1]

		local CriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

		local yValue = rewardValue + (NewDeepDeterministicPolicyGradientModel.discountFactor * (1 - terminalStateValue) * targetQValue)

		local actionTensor = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		actionTensor = AqwamTensorLibrary:add(actionTensor, previousActionMeanTensor)

		local previousCriticActionInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, actionTensor, 2)

		local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputTensor, true)[1][1]

		local negatedtemporalDifferenceError = currentQValue - yValue
		
		local temporalDifferenceError = -negatedtemporalDifferenceError

		ActorModel:forwardPropagate(previousFeatureTensor, true)

		ActorModel:update(negatedtemporalDifferenceError, true)

		local previousCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, previousActionMeanTensor, 2)

		CriticModel:forwardPropagate(previousCriticActionMeanInputTensor, true)

		CriticModel:update(temporalDifferenceError, true)

		local TargetActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local TargetCriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

		TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, ActorWeightTensorArray)

		TargetCriticWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArray, CriticWeightTensorArray)

		ActorModel:setWeightTensorArray(TargetActorWeightTensorArray, true)

		CriticModel:setWeightTensorArray(TargetCriticWeightTensorArray, true)

		return temporalDifferenceError

	end)

	NewDeepDeterministicPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue) end)

	NewDeepDeterministicPolicyGradientModel:setResetFunction(function() end)

	return NewDeepDeterministicPolicyGradientModel

end

return DeepDeterministicPolicyGradientModel
