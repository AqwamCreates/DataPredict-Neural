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

RecurrentDeepDoubleQLearningModel = {}

RecurrentDeepDoubleQLearningModel.__index = RecurrentDeepDoubleQLearningModel

setmetatable(RecurrentDeepDoubleQLearningModel, RecurrentReinforcementLearningBaseModel)

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

function RecurrentDeepDoubleQLearningModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepDoubleQLearningModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepDoubleQLearningModel, RecurrentDeepDoubleQLearningModel)

	NewRecurrentDeepDoubleQLearningModel:setName("RecurrentDeepDoubleQLearning")

	NewRecurrentDeepDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewRecurrentDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepDoubleQLearningModel.Model

		local discountFactor = NewRecurrentDeepDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewRecurrentDeepDoubleQLearningModel.EligibilityTrace
		
		local hiddenStateTensor = NewRecurrentDeepDoubleQLearningModel.hiddenStateTensor

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end
		
		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local _, maxQValue = Model:predict(currentFeatureTensor, previousQTensor)

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousQTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		local TargetWeightTensorArray = Model:getWeightTensorArray(true)

		TargetWeightTensorArray = rateAverageWeightTensorArray(NewRecurrentDeepDoubleQLearningModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		Model:setWeightTensorArray(TargetWeightTensorArray, true)
		
		NewRecurrentDeepDoubleQLearningModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceError

	end)

	NewRecurrentDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		NewRecurrentDeepDoubleQLearningModel.EligibilityTrace:reset()

	end)

	NewRecurrentDeepDoubleQLearningModel:setResetFunction(function() 

		NewRecurrentDeepDoubleQLearningModel.EligibilityTrace:reset()

	end)

	return NewRecurrentDeepDoubleQLearningModel

end

return RecurrentDeepDoubleQLearningModel
