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

local DeepExpectedStateActionRewardStateActionModel = {}

DeepExpectedStateActionRewardStateActionModel.__index = DeepExpectedStateActionRewardStateActionModel

setmetatable(DeepExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

function DeepExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepExpectedStateActionRewardStateActionModel, DeepExpectedStateActionRewardStateActionModel)
	
	NewDeepExpectedStateActionRewardStateActionModel:setName("DeepExpectedStateActionRewardStateAction")
	
	NewDeepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewDeepExpectedStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewDeepExpectedStateActionRewardStateActionModel.epsilon
		
		local EligibilityTrace = NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local actionIndex = table.find(ClassesList, previousAction)
		
		local currentQTensor = Model:forwardPropagate(currentFeatureTensor)

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor, true)

		local maximumCurrentQValue = AqwamTensorLibrary:findMaximumValue(currentQTensor)

		local unwrappedTargetTensor = currentQTensor[1]

		for i = 1, numberOfClasses, 1 do

			if (unwrappedTargetTensor[i] == maximumCurrentQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		local actionProbability

		for _, qValue in ipairs(unwrappedTargetTensor) do

			actionProbability = ((qValue == maximumCurrentQValue) and greedyActionProbability) or nonGreedyActionProbability

			expectedQValue = expectedQValue + (qValue * actionProbability)

		end
		
		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local previousQValue = previousQTensor[1][actionIndex]

		local temporalDifferenceError = targetQValue - previousQValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)
		
		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepExpectedStateActionRewardStateActionModel

end

return DeepExpectedStateActionRewardStateActionModel
