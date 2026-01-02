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

local RecurrentDeepNStepStateActionRewardStateActionModel = {}

RecurrentDeepNStepStateActionRewardStateActionModel.__index = RecurrentDeepNStepStateActionRewardStateActionModel

setmetatable(RecurrentDeepNStepStateActionRewardStateActionModel, RecurrentReinforcementLearningBaseModel)

local defaultNStep = 3

function RecurrentDeepNStepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepNStepStateActionRewardStateActionModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewRecurrentDeepNStepStateActionRewardStateActionModel, RecurrentDeepNStepStateActionRewardStateActionModel)
	
	NewRecurrentDeepNStepStateActionRewardStateActionModel:setName("RecurrentDeepNStepStateActionRewardStateAction")
	
	NewRecurrentDeepNStepStateActionRewardStateActionModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewRecurrentDeepNStepStateActionRewardStateActionModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewRecurrentDeepNStepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local nStep = NewRecurrentDeepNStepStateActionRewardStateActionModel.nStep

		local replayBufferArray = NewRecurrentDeepNStepStateActionRewardStateActionModel.replayBufferArray

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

		local Model = NewRecurrentDeepNStepStateActionRewardStateActionModel.Model

		local discountFactor = NewRecurrentDeepNStepStateActionRewardStateActionModel.discountFactor

		local ClassesList = Model:getClassesList()

		local returnValue = 0

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
		
		local currentActionIndex = table.find(ClassesList, currentAction)
		
		local previousActionIndex = table.find(ClassesList, firstExperience[2])

		local bootstrapValue = math.pow(discountFactor, currentNStep) * currentQTensor[1][currentActionIndex]	

		local nStepTarget = returnValue + bootstrapValue

		local lastValue = lastQTensor[1][previousActionIndex]

		local temporalDifferenceError = nStepTarget - lastValue

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		negatedTemporalDifferenceErrorTensor[1][previousActionIndex] = -temporalDifferenceError -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
		
		Model:forwardPropagate(firstExperience[1], hiddenTensor, true)
		
		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		return temporalDifferenceError

	end)
	
	NewRecurrentDeepNStepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		table.clear(NewRecurrentDeepNStepStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	NewRecurrentDeepNStepStateActionRewardStateActionModel:setResetFunction(function()
		
		table.clear(NewRecurrentDeepNStepStateActionRewardStateActionModel.replayBufferArray)
		
	end)

	return NewRecurrentDeepNStepStateActionRewardStateActionModel

end

return RecurrentDeepNStepStateActionRewardStateActionModel
