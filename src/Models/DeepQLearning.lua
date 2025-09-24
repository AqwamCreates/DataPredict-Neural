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

function DeepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepQLearningModel, DeepQLearningModel)
	
	NewDeepQLearningModel:setName("DeepQLearning")
	
	NewDeepQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepQLearningModel.Model
		
		local discountFactor = NewDeepQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepQLearningModel.EligibilityTrace

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
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, true)

		Model:update(negatedTemporalDifferenceErrorTensor, true)

		return temporalDifferenceError

	end)

	NewDeepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepQLearningModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepQLearningModel

end

return DeepQLearningModel
