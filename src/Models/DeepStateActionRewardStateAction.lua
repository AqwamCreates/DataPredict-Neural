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

DeepStateActionRewardStateActionModel = {}

DeepStateActionRewardStateActionModel.__index = DeepStateActionRewardStateActionModel

setmetatable(DeepStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

function DeepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepStateActionRewardStateActionModel, DeepStateActionRewardStateActionModel)
	
	NewDeepStateActionRewardStateActionModel:setName("DeepStateActionRewardStateAction")
	
	NewDeepStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewDeepStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewDeepStateActionRewardStateActionModel.EligibilityTrace

		local currentQTensor = Model:forwardPropagate(currentFeatureTensor)

		local previousQTensor = Model:forwardPropagate(previousFeatureTensor)

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local previousActionIndex = table.find(ClassesList, previousAction)

		local currentActionIndex = table.find(ClassesList, currentAction)

		local targetValue = rewardValue + (discountFactor * currentQTensor[1][currentActionIndex] * (1 - terminalStateValue))

		local temporalDifferenceError = targetValue - previousQTensor[1][previousActionIndex] 

		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorTensor[1][previousActionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(previousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorTensor = EligibilityTrace:calculate(temporalDifferenceErrorTensor)

		end

		local negatedTemporalDifferenceErrorTensor = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorTensor) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureTensor)

		Model:update(negatedTemporalDifferenceErrorTensor)

		return temporalDifferenceError

	end)

	NewDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepStateActionRewardStateActionModel

end

return DeepStateActionRewardStateActionModel
