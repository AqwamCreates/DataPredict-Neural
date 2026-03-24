--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local RecurrentReinforcementLearningBaseModel = require(script.Parent.RecurrentReinforcementLearningBaseModel)

RecurrentDeepQLearningModel = {}

RecurrentDeepQLearningModel.__index = RecurrentDeepQLearningModel

setmetatable(RecurrentDeepQLearningModel, RecurrentReinforcementLearningBaseModel)

function RecurrentDeepQLearningModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepQLearningModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepQLearningModel, RecurrentDeepQLearningModel)

	NewRecurrentDeepQLearningModel:setName("RecurrentDeepQLearning")

	NewRecurrentDeepQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentDeepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewRecurrentDeepQLearningModel.Model

		local discountFactor = NewRecurrentDeepQLearningModel.discountFactor
		
		local EligibilityTrace = NewRecurrentDeepQLearningModel.EligibilityTrace

		local hiddenStateTensor = NewRecurrentDeepQLearningModel.hiddenStateTensor

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local outputDimensionSizeArray = {1, numberOfClasses}

		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)
		
		local _, maxQValue = Model:predict(currentFeatureTensor, previousQTensor)

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

		local previousActionIndex = table.find(ClassesList, previousAction)

		local previousQValue = previousQTensor[1][previousActionIndex]

		local temporalDifferenceError = targetQValue - previousQValue

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][previousActionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(previousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		NewRecurrentDeepQLearningModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceError

	end)

	NewRecurrentDeepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		local EligibilityTrace = NewRecurrentDeepQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepQLearningModel.hiddenStateTensor = nil

	end)

	NewRecurrentDeepQLearningModel:setResetFunction(function() 

		local EligibilityTrace = NewRecurrentDeepQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		NewRecurrentDeepQLearningModel.hiddenStateTensor = nil

	end)

	return NewRecurrentDeepQLearningModel

end

return RecurrentDeepQLearningModel
