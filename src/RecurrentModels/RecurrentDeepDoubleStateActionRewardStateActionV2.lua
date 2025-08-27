--[[

	--------------------------------------------------------------------

	Aqwam's RecurrentDeep Learning Library (DataPredict Neural)

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

RecurrentDeepDoubleStateActionRewardStateActionModel = {}

RecurrentDeepDoubleStateActionRewardStateActionModel.__index = RecurrentDeepDoubleStateActionRewardStateActionModel

setmetatable(RecurrentDeepDoubleStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleStateActionRewardStateActionModel, RecurrentDeepDoubleStateActionRewardStateActionModel)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setName("RecurrentDeepDoubleStateActionRewardStateAction")

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepDoubleStateActionRewardStateActionModel.Model

		local discountFactor = NewRecurrentDeepDoubleStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local hiddenStateTensor = NewRecurrentDeepDoubleStateActionRewardStateActionModel.hiddenStateTensor
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end
		
		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local qTensor = Model:forwardPropagate(currentFeatureTensor, previousQTensor)

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		local discountedQTensor = AqwamTensorLibrary:multiply(discountFactor, qTensor, (1 - terminalStateValue))

		local targetTensor = AqwamTensorLibrary:add(rewardValue, discountedQTensor)

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:subtract(targetTensor, previousQTensor)

		if (EligibilityTrace) then

			local ClassesList = Model:getClassesList()

			local actionIndex = table.find(ClassesList, action)

			EligibilityTrace:increment(actionIndex, discountFactor, {1, #ClassesList})

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		local TargetWeightTensorArray = Model:getWeightTensorArray(true)

		TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		Model:setWeightTensorArray(TargetWeightTensorArray, true)

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceErrorTensor

	end)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	NewRecurrentDeepDoubleStateActionRewardStateActionModel:setResetFunction(function() 

		NewRecurrentDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	return NewRecurrentDeepDoubleStateActionRewardStateActionModel

end

return RecurrentDeepDoubleStateActionRewardStateActionModel
