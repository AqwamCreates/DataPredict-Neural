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

RecurrentDeepExpectedStateActionRewardStateActionModel = {}

RecurrentDeepExpectedStateActionRewardStateActionModel.__index = RecurrentDeepExpectedStateActionRewardStateActionModel

setmetatable(RecurrentDeepExpectedStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

function RecurrentDeepExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepExpectedStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewRecurrentDeepExpectedStateActionRewardStateActionModel, RecurrentDeepExpectedStateActionRewardStateActionModel)

	NewRecurrentDeepExpectedStateActionRewardStateActionModel:setName("RecurrentDeepExpectedStateActionRewardStateAction")

	NewRecurrentDeepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewRecurrentDeepExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewRecurrentDeepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewRecurrentDeepExpectedStateActionRewardStateActionModel.Model

		local epsilon = NewRecurrentDeepExpectedStateActionRewardStateActionModel.epsilon

		local discountFactor = NewRecurrentDeepExpectedStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewRecurrentDeepExpectedStateActionRewardStateActionModel.EligibilityTrace
		
		local hiddenStateTensor = NewRecurrentDeepExpectedStateActionRewardStateActionModel.hiddenStateTensor
		
		local ClassesList = Model:getClassesList()
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}
		
		if (not hiddenStateTensor) then hiddenStateTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray) end

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local actionIndex = table.find(ClassesList, action)

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		local targetTensor = Model:forwardPropagate(currentFeatureTensor, previousQTensor)

		local maxQValue = targetTensor[1][actionIndex]

		for i = 1, numberOfClasses, 1 do

			if (targetTensor[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(targetTensor[1]) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local lastValue = previousQTensor[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor, hiddenStateTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)
		
		NewRecurrentDeepExpectedStateActionRewardStateActionModel.hiddenStateTensor = previousQTensor

		return temporalDifferenceError

	end)

	NewRecurrentDeepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 

		NewRecurrentDeepExpectedStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	NewRecurrentDeepExpectedStateActionRewardStateActionModel:setResetFunction(function() 

		NewRecurrentDeepExpectedStateActionRewardStateActionModel.EligibilityTrace:reset()

	end)

	return NewRecurrentDeepExpectedStateActionRewardStateActionModel

end

return RecurrentDeepExpectedStateActionRewardStateActionModel
