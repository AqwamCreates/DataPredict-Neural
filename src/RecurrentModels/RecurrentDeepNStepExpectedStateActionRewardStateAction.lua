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

local RecurrentReinforcementLearningBaseModel = require(script.Parent.RecurrentReinforcementLearningBaseModel)

local RecurrentDeepNStepExpectedStateActionRewardStateActionModel = {}

RecurrentDeepNStepExpectedStateActionRewardStateActionModel.__index = RecurrentDeepNStepExpectedStateActionRewardStateActionModel

setmetatable(RecurrentDeepNStepExpectedStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultNStep = 3

function RecurrentDeepNStepExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel, RecurrentDeepNStepExpectedStateActionRewardStateActionModel)
	
	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel:setName("RecurrentDeepNStepExpectedStateActionRewardStateAction")
	
	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local nStep = NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.nStep

		local replayBufferArray = NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray

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

		local Model = NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.Model

		local discountFactor = NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.epsilon

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
		
		local outputDimensionSizeArray = {1, #ClassesList}

		local hiddenTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		local previousFeatureTensor

		local qTensor

		local lastQTensor

		local currentQTensor

		for i, experience in ipairs(replayBufferArray) do

			previousFeatureTensor = experience[1]

			qTensor = Model:forwardPropagate(previousFeatureTensor, hiddenTensor)

			hiddenTensor = qTensor

			if (i == 1) then

				lastQTensor = qTensor

			elseif (i == nStep) then

				currentQTensor = qTensor

			end

		end

		hiddenTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		local firstExperience = replayBufferArray[1]

		local maxQValue = AqwamTensorLibrary:findMaximumValue(currentQTensor)

		local actionIndex = table.find(ClassesList, previousAction)

		local unwrappedTargetTensor = currentQTensor[1]

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

		local lastValue = lastQTensor[1][actionIndex]

		local temporalDifferenceError = nStepTarget - lastValue

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		negatedTemporalDifferenceErrorTensor[1][actionIndex] = -temporalDifferenceError -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:forwardPropagate(firstExperience[1], hiddenTensor, true)
		
		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		return temporalDifferenceError

	end)
	
	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		table.clear(NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel:setResetFunction(function()
		
		table.clear(NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	return NewRecurrentDeepNStepExpectedStateActionRewardStateActionModel

end

return RecurrentDeepNStepExpectedStateActionRewardStateActionModel
