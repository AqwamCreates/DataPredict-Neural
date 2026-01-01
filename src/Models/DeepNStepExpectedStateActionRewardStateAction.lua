--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

local DeepNStepExpectedStateActionRewardStateActionModel = {}

DeepNStepExpectedStateActionRewardStateActionModel.__index = DeepNStepExpectedStateActionRewardStateActionModel

setmetatable(DeepNStepExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultNStep = 3

function DeepNStepExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepNStepExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepNStepExpectedStateActionRewardStateActionModel, DeepNStepExpectedStateActionRewardStateActionModel)
	
	NewDeepNStepExpectedStateActionRewardStateActionModel:setName("DeepNStepExpectedStateActionRewardStateAction")
	
	NewDeepNStepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewDeepNStepExpectedStateActionRewardStateActionModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewDeepNStepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local nStep = NewDeepNStepExpectedStateActionRewardStateActionModel.nStep

		local replayBufferArray = NewDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray

		table.insert(replayBufferArray, {previousFeatureTensor, previousAction, rewardValue, terminalStateValue})

		local currentNStep = #replayBufferArray

		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end

		if (currentNStep > nStep) then 

			table.remove(replayBufferArray, 1)

			currentNStep = currentNStep - 1

		end
		
		if (currentNStep < nStep) and (terminalStateValue == 0) then return 0 end

		if (currentNStep > nStep) then 

			table.remove(replayBufferArray, 1)

			currentNStep = currentNStep - 1

		end

		local Model = NewDeepNStepExpectedStateActionRewardStateActionModel.Model

		local discountFactor = NewDeepNStepExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewDeepNStepExpectedStateActionRewardStateActionModel.epsilon

		local ClassesList = Model:getClassesList()
		
		local numberOfClasses = #ClassesList

		local returnValue = 0
		
		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local experience

		local rewardValueAtStepI

		local terminalStateValueAtStepI

		for i = currentNStep, 1, -1 do

			experience = replayBufferArray[i]

			rewardValueAtStepI = experience[3]

			terminalStateValueAtStepI = experience[4]

			returnValue = rewardValueAtStepI + (discountFactor * (1 - terminalStateValueAtStepI) * returnValue)

		end

		local firstExperience = replayBufferArray[1]

		local targetTensor = Model:forwardPropagate(currentFeatureTensor)

		local previousTensor = Model:forwardPropagate(previousFeatureTensor, true)

		local maxQValue = AqwamTensorLibrary:findMaximumValue(targetTensor)

		local actionIndex = table.find(ClassesList, previousAction)

		local unwrappedTargetTensor = targetTensor[1]

		for i = 1, numberOfClasses, 1 do

			if (unwrappedTargetTensor[i] == maxQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		local actionProbability

		for _, qValue in ipairs(unwrappedTargetTensor) do

			actionProbability = ((qValue == maxQValue) and greedyActionProbability) or nonGreedyActionProbability

			expectedQValue = expectedQValue + (qValue * actionProbability)

		end

		local bootstrapValue = math.pow(discountFactor, currentNStep) * expectedQValue

		local nStepTarget = returnValue + bootstrapValue

		local lastValue = previousTensor[1][actionIndex]

		local temporalDifferenceError = nStepTarget - lastValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		negatedTemporalDifferenceErrorTensor[1][actionIndex] = -temporalDifferenceError -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepNStepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		table.clear(NewDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	NewDeepNStepExpectedStateActionRewardStateActionModel:setResetFunction(function()
		
		table.clear(NewDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	return NewDeepNStepExpectedStateActionRewardStateActionModel

end

return DeepNStepExpectedStateActionRewardStateActionModel
