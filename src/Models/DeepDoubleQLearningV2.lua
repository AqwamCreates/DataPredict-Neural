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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, ReinforcementLearningBaseModel)

local defaultAveragingRate = 0.995

local defaultLambda = 0

local function rateAverageWeightTensorArray(averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetWeightTensorArray, 1 do

		local TargetWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRate, TargetWeightTensorArray[layer])

		local PrimaryWeightTensorArrayPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryWeightTensorArray[layer])

		TargetWeightTensorArray[layer] = AqwamTensorLibrary:add(TargetWeightTensorArrayPart, PrimaryWeightTensorArrayPart)

	end

	return TargetWeightTensorArray

end

function DeepDoubleQLearningModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearning")

	NewDeepDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDoubleQLearningModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewDeepDoubleQLearningModel.eligibilityTraceTensor = parameterDictionary.eligibilityTraceTensor

	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepDoubleQLearningModel.Model
		
		local lambda = NewDeepDoubleQLearningModel.lambda
		
		local discountFactor = NewDeepDoubleQLearningModel.discountFactor
		
		local _, maxQValue = Model:predict(currentFeatureTensor)

		local PrimaryWeightTensorArray = Model:getWeightTensorArray(true)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor)

		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousQTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
		
		if (lambda ~= 0) then

			local eligibilityTraceTensor = NewDeepDoubleQLearningModel.eligibilityTraceTensor

			if (not eligibilityTraceTensor) then eligibilityTraceTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

			eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * lambda)

			eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

			temporalDifferenceErrorTensor = AqwamTensorLibrary:multiply(temporalDifferenceErrorTensor, eligibilityTraceTensor)

			NewDeepDoubleQLearningModel.eligibilityTraceTensor = eligibilityTraceTensor

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		local TargetWeightTensorArray = Model:getWeightTensorArray(true)

		TargetWeightTensorArray = rateAverageWeightTensorArray(NewDeepDoubleQLearningModel.averagingRate, TargetWeightTensorArray, PrimaryWeightTensorArray)

		Model:setWeightTensorArray(TargetWeightTensorArray, true)

		return temporalDifferenceError

	end)

	NewDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleQLearningModel.eligibilityTraceTensor = nil
		
	end)

	NewDeepDoubleQLearningModel:setResetFunction(function() 
		
		NewDeepDoubleQLearningModel.eligibilityTraceTensor = nil
		
	end)

	return NewDeepDoubleQLearningModel

end

return DeepDoubleQLearningModel