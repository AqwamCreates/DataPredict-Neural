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

local DualRecurrentReinforcementLearningBaseModel = require(script.Parent.DualRecurrentReinforcementLearningBaseModel)

DeepClippedDoubleQLearningModel = {}

DeepClippedDoubleQLearningModel.__index = DeepClippedDoubleQLearningModel

setmetatable(DeepClippedDoubleQLearningModel, DualRecurrentReinforcementLearningBaseModel)

function DeepClippedDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepClippedDoubleQLearningModel = DualRecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepClippedDoubleQLearningModel, DeepClippedDoubleQLearningModel)
	
	NewDeepClippedDoubleQLearningModel:setName("DeepClippedDoubleQLearning")
	
	NewDeepClippedDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepClippedDoubleQLearningModel.WeightTensorArrayArray = {}

	NewDeepClippedDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

		local Model = NewDeepClippedDoubleQLearningModel.Model
		
		local discountFactor = NewDeepClippedDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepClippedDoubleQLearningModel.EligibilityTrace
		
		local WeightTensorArrayArray = NewDeepClippedDoubleQLearningModel.WeightTensorArrayArray
		
		local hiddenStateTensorArray = NewDeepClippedDoubleQLearningModel.hiddenStateTensorArray

		local maxQValueArray = {}
		
		local ClassesList = Model:getClassesList()
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}
		
		hiddenStateTensorArray[1] = hiddenStateTensorArray[1] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)
		
		hiddenStateTensorArray[2] = hiddenStateTensorArray[2] or AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		for i = 1, 2, 1 do

			Model:setWeightTensorArray(WeightTensorArrayArray[i], true)
			
			local previousTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[i])

			local _, maxQValue = Model:predict(currentFeatureTensor, previousTensor)

			table.insert(maxQValueArray, maxQValue[1][1])
			
			WeightTensorArrayArray[i] = Model:getWeightTensorArray(true)

		end

		local maxQValue = math.min(table.unpack(maxQValueArray))

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)

		local actionIndex = table.find(ClassesList, previousAction)
		
		local eligibilityTraceTensor = NewDeepClippedDoubleQLearningModel.eligibilityTraceTensor

		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2})
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

		end

		for i = 1, 2, 1 do

			Model:setWeightTensorArray(WeightTensorArrayArray[i], true)

			local previousTensor = Model:forwardPropagate(previousFeatureTensor, hiddenStateTensorArray[i])

			local lastValue = previousTensor[1][actionIndex]

			local temporalDifferenceError = targetValue - lastValue

			local lossTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)

			lossTensor[1][actionIndex] = temporalDifferenceError

			temporalDifferenceErrorTensor[1][i] = temporalDifferenceError
			
			if (EligibilityTrace) then lossTensor = EligibilityTrace:calculate(lossTensor) end
			
			local negatedLossTensor = AqwamTensorLibrary:unaryMinus(lossTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error tensor to make the neural network to perform gradient ascent.

			Model:update(negatedLossTensor)
			
			WeightTensorArrayArray[i] = Model:getWeightTensorArray(true)
			
			hiddenStateTensorArray[i] = previousTensor

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
