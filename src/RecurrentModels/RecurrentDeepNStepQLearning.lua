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

local RecurrentDeepNStepQLearningModel = {}

RecurrentDeepNStepQLearningModel.__index = RecurrentDeepNStepQLearningModel

setmetatable(RecurrentDeepNStepQLearningModel, RecurrentReinforcementLearningBaseModel)

local defaultNStep = 3

function RecurrentDeepNStepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRecurrentDeepNStepQLearningModel = RecurrentReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewRecurrentDeepNStepQLearningModel, RecurrentDeepNStepQLearningModel)
	
	NewRecurrentDeepNStepQLearningModel:setName("RecurrentDeepNStepQLearning")
	
	NewRecurrentDeepNStepQLearningModel.nStep = parameterDictionary.nStep or defaultNStep
	
	NewRecurrentDeepNStepQLearningModel.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewRecurrentDeepNStepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local nStep = NewRecurrentDeepNStepQLearningModel.nStep

		local replayBufferArray = NewRecurrentDeepNStepQLearningModel.replayBufferArray

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

		local Model = NewRecurrentDeepNStepQLearningModel.Model

		local discountFactor = NewRecurrentDeepNStepQLearningModel.discountFactor

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
		
		local currentMaxQValue
		
		local currentHiddenTensor
		
		for i, experience in ipairs(replayBufferArray) do
			
			previousFeatureTensor = experience[1]
			
			qTensor = Model:forwardPropagate(previousFeatureTensor, hiddenTensor, true)
			
			if (i == 1) then
				
				lastQTensor = qTensor
				
			elseif (i == nStep) then
				
				currentQTensor = qTensor
				
				currentHiddenTensor = hiddenTensor
				
				currentMaxQValue = AqwamTensorLibrary:findMaximumValue(currentQTensor)
				
			end
			
		end
		
		hiddenTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)
		
		local firstExperience = replayBufferArray[1]

		local bootstrapValue = math.pow(discountFactor, currentNStep) * currentMaxQValue

		local nStepTarget = returnValue + bootstrapValue

		local actionIndex = table.find(ClassesList, firstExperience[2])

		local lastValue = lastQTensor[1][actionIndex]

		local temporalDifferenceError = nStepTarget - lastValue
		
		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][actionIndex] = temporalDifferenceError
		
		Model:forwardPropagate(firstExperience[1], hiddenTensor, true)
		
		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.

		Model:update(negatedTemporalDifferenceErrorTensor, true)
		
		return temporalDifferenceErrorTensor

	end)
	
	NewRecurrentDeepNStepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		table.clear(NewRecurrentDeepNStepQLearningModel.replayBufferArray)
		
	end)

	NewRecurrentDeepNStepQLearningModel:setResetFunction(function()
		
		table.clear(NewRecurrentDeepNStepQLearningModel.replayBufferArray)
		
	end)

	return NewRecurrentDeepNStepQLearningModel

end

return RecurrentDeepNStepQLearningModel
