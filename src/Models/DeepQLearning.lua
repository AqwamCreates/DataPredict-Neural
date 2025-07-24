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

DeepQLearningModel = {}

DeepQLearningModel.__index = DeepQLearningModel

setmetatable(DeepQLearningModel, ReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepQLearningModel, DeepQLearningModel)
	
	NewDeepQLearningModel:setName("DeepQLearning")
	
	NewDeepQLearningModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewDeepQLearningModel.eligibilityTraceTensor = parameterDictionary.eligibilityTraceTensor

	NewDeepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepQLearningModel.Model
		
		local lambda = NewDeepQLearningModel.lambda
		
		local discountFactor = NewDeepQLearningModel.discountFactor

		local predictedValue, maxQValue = Model:predict(currentFeatureTensor)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local previousTensor = Model:forwardPropagate(previousFeatureTensor)

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
		
		if (lambda ~= 0) then

			local ClassesList = Model:getClassesList()

			local actionIndex = table.find(ClassesList, action)

			local eligibilityTraceTensor = NewDeepQLearningModel.eligibilityTraceTensor

			if (not eligibilityTraceTensor) then eligibilityTraceTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

			eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * lambda)

			eligibilityTraceTensor[1][actionIndex] = eligibilityTraceTensor[1][actionIndex] + 1

			temporalDifferenceErrorTensor = AqwamTensorLibrary:multiply(temporalDifferenceErrorTensor, eligibilityTraceTensor)

			NewDeepQLearningModel.eligibilityTraceTensor = eligibilityTraceTensor

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		return temporalDifferenceError

	end)

	NewDeepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepQLearningModel.eligibilityTraceTensor = nil
		
	end)

	NewDeepQLearningModel:setResetFunction(function() 
		
		NewDeepQLearningModel.eligibilityTraceTensor = nil
		
	end)

	return NewDeepQLearningModel

end

return DeepQLearningModel