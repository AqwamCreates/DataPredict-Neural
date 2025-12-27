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

RecurrentDeepDeterministicPolicyGradientModel = {}

RecurrentDeepDeterministicPolicyGradientModel.__index = RecurrentDeepDeterministicPolicyGradientModel

setmetatable(RecurrentDeepDeterministicPolicyGradientModel, DualRecurrentReinforcementLearningActorCriticBaseModel)

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

function RecurrentDeepDeterministicPolicyGradientModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDeterministicPolicyGradientModel = DualRecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDeterministicPolicyGradientModel, RecurrentDeepDeterministicPolicyGradientModel)

	NewRecurrentDeepDeterministicPolicyGradientModel:setName("RecurrentDeepDeterministicPolicyGradient")

	NewRecurrentDeepDeterministicPolicyGradientModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDeterministicPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

		if (not previousActionNoiseTensor) then

			local actionTensordimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

			previousActionNoiseTensor = AqwamTensorLibrary:createRandomUniformTensor(actionTensordimensionSizeArray) 

		end

		local ActorModel = NewRecurrentDeepDeterministicPolicyGradientModel.ActorModel

		local CriticModel = NewRecurrentDeepDeterministicPolicyGradientModel.CriticModel

		local averagingRate = NewRecurrentDeepDeterministicPolicyGradientModel.averagingRate
		
		local actorHiddenStateTensorArray = NewRecurrentDeepDeterministicPolicyGradientModel.actorHiddenStateTensorArray

		local criticHiddenStateValueArray = NewRecurrentDeepDeterministicPolicyGradientModel.criticHiddenStateValueArray

		local outputDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(previousActionMeanTensor)

		local currentActorHiddenStateTensor = actorHiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

		local targetActorHiddenStateTensor = actorHiddenStateTensorArray[2] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		local currentCriticHiddenStateTensor = criticHiddenStateValueArray[1] or {{0}}
		
		local targetCriticHiddenStateTensor = criticHiddenStateValueArray[2] or {{0}}
		
		local previousActionMeanTensor = ActorModel:forwardPropagate(previousFeatureTensor, targetActorHiddenStateTensor)

		local currentActionMeanTensor = ActorModel:forwardPropagate(currentFeatureTensor, currentActorHiddenStateTensor)

		local ActorWeightTensorArray = ActorModel:getWeightTensorArray(true)
		
		local previousTargetCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, previousActionMeanTensor, 2)

		local targetCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(currentFeatureTensor, currentActionMeanTensor, 2)
		
		local previousTargetQTensor = CriticModel:forwardPropagate(previousTargetCriticActionMeanInputTensor, targetCriticHiddenStateTensor)

		local targetQValue = CriticModel:forwardPropagate(targetCriticActionMeanInputTensor, previousTargetQTensor)[1][1]

		local CriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

		local yValue = rewardValue + (NewRecurrentDeepDeterministicPolicyGradientModel.discountFactor * (1 - terminalStateValue) * targetQValue)

		local actionTensor = AqwamTensorLibrary:multiply(previousActionStandardDeviationTensor, previousActionNoiseTensor)

		actionTensor = AqwamTensorLibrary:add(actionTensor, previousActionMeanTensor)

		local previousCriticActionInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, actionTensor, 2)

		local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputTensor, currentCriticHiddenStateTensor)[1][1]

		local negatedtemporalDifferenceError = currentQValue - yValue
		
		local temporalDifferenceError = -negatedtemporalDifferenceError

		local previousActorHiddenStateTensor = ActorModel:forwardPropagate(previousFeatureTensor, currentActorHiddenStateTensor)

		ActorModel:update(negatedtemporalDifferenceError)

		local previousCriticActionMeanInputTensor = AqwamTensorLibrary:concatenate(previousFeatureTensor, previousActionMeanTensor, 2)

		local previousCriticHiddenStateTensor = CriticModel:forwardPropagate(previousCriticActionMeanInputTensor, currentCriticHiddenStateTensor)

		CriticModel:update(temporalDifferenceError, true)

		local TargetActorWeightTensorArray = ActorModel:getWeightTensorArray(true)

		local TargetCriticWeightTensorArray = CriticModel:getWeightTensorArray(true)

		TargetActorWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetActorWeightTensorArray, ActorWeightTensorArray)

		TargetCriticWeightTensorArray = rateAverageWeightTensorArray(averagingRate, TargetCriticWeightTensorArray, CriticWeightTensorArray)

		ActorModel:setWeightTensorArray(TargetActorWeightTensorArray, true)

		CriticModel:setWeightTensorArray(TargetCriticWeightTensorArray, true)
		
		actorHiddenStateTensorArray[1] = previousActorHiddenStateTensor
		
		actorHiddenStateTensorArray[2] = previousActionMeanTensor
		
		criticHiddenStateValueArray[1] = previousCriticHiddenStateTensor
		
		criticHiddenStateValueArray[2] = previousTargetQTensor

		return temporalDifferenceError

	end)

	NewRecurrentDeepDeterministicPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue) end)

	NewRecurrentDeepDeterministicPolicyGradientModel:setResetFunction(function() end)

	return NewRecurrentDeepDeterministicPolicyGradientModel

end

return RecurrentDeepDeterministicPolicyGradientModel
