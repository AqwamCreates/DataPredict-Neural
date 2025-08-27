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

DeepClippedDoubleQLearningModel = {}

DeepClippedDoubleQLearningModel.__index = DeepClippedDoubleQLearningModel

setmetatable(DeepClippedDoubleQLearningModel, ReinforcementLearningBaseModel)

function DeepClippedDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepClippedDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepClippedDoubleQLearningModel, DeepClippedDoubleQLearningModel)
	
	NewDeepClippedDoubleQLearningModel:setName("DeepClippedDoubleQLearning")
	
	NewDeepClippedDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepClippedDoubleQLearningModel.WeightTensorArrayArray = {}

	NewDeepClippedDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, action, rewardValue, currentFeatureTensor, terminalStateValue)

		local Model = NewDeepClippedDoubleQLearningModel.Model
		
		local discountFactor = NewDeepClippedDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepClippedDoubleQLearningModel.EligibilityTrace
		
		local WeightTensorArrayArray = NewDeepClippedDoubleQLearningModel.WeightTensorArrayArray
		
		local maxQValueArray = {}

		for i = 1, 2, 1 do

			Model:setWeightTensorArray(WeightTensorArrayArray[i], true)

			local _, maxQValue = Model:predict(currentFeatureTensor)

			table.insert(maxQValueArray, maxQValue[1][1])
			
			WeightTensorArrayArray[i] = Model:getWeightTensorArray(true)

		end

		local maxQValue = math.min(table.unpack(maxQValueArray))

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)

		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2})
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

		end

		for i = 1, 2, 1 do

			Model:setWeightTensorArray(WeightTensorArrayArray[i], true)

			local previousTensor = Model:forwardPropagate(previousFeatureTensor, true)

			local lastValue = previousTensor[1][actionIndex]

			local temporalDifferenceError = targetValue - lastValue

			local lossTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

			lossTensor[1][actionIndex] = temporalDifferenceError

			temporalDifferenceErrorTensor[1][i] = temporalDifferenceError
			
			if (EligibilityTrace) then lossTensor = EligibilityTrace:calculate(lossTensor) end
			
			local negatedLossTensor = AqwamTensorLibrary:unaryMinus(lossTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

			Model:update(negatedLossTensor, true)
			
			WeightTensorArrayArray[i] = Model:getWeightTensorArray(true)

		end

		return temporalDifferenceErrorTensor

	end)

	NewDeepClippedDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepClippedDoubleQLearningModel.EligibilityTrace:reset()
		
	end)

	NewDeepClippedDoubleQLearningModel:setResetFunction(function() 
		
		NewDeepClippedDoubleQLearningModel.EligibilityTrace:reset()
		
	end)

	return NewDeepClippedDoubleQLearningModel

end

function DeepClippedDoubleQLearningModel:setWeightTensorArray1(WeightTensorArray1)

	self.WeightTensorArrayArray[1] = WeightTensorArray1

end

function DeepClippedDoubleQLearningModel:setWeightTensorArray2(WeightTensorArray2)

	self.WeightTensorArrayArray[2] = WeightTensorArray2

end

function DeepClippedDoubleQLearningModel:getWeightTensorArray1(WeightTensorArray1)

	return self.WeightTensorArrayArray[1]

end

function DeepClippedDoubleQLearningModel:getWeightTensorArray2(WeightTensorArray2)

	return self.WeightTensorArrayArray[2]

end

return DeepClippedDoubleQLearningModel
