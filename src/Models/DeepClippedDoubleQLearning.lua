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

local DeepClippedDoubleQLearningModel = {}

DeepClippedDoubleQLearningModel.__index = DeepClippedDoubleQLearningModel

setmetatable(DeepClippedDoubleQLearningModel, ReinforcementLearningBaseModel)

function DeepClippedDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepClippedDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepClippedDoubleQLearningModel, DeepClippedDoubleQLearningModel)
	
	NewDeepClippedDoubleQLearningModel:setName("DeepClippedDoubleQLearning")
	
	NewDeepClippedDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepClippedDoubleQLearningModel.WeightTensorArrayArray = {}
	
	NewDeepClippedDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)
		
		local Model = NewDeepClippedDoubleQLearningModel.Model
		
		local discountFactor = NewDeepClippedDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepClippedDoubleQLearningModel.EligibilityTrace
		
		local WeightTensorArrayArray = NewDeepClippedDoubleQLearningModel.WeightTensorArrayArray

		local maximumCurrentQValueArray = {}

		for i = 1, 2, 1 do

			Model:setWeightTensorArray(WeightTensorArrayArray[i], true)

			local _, maximumCurrentQTensor = Model:predict(currentFeatureTensor)

			table.insert(maximumCurrentQValueArray, maximumCurrentQTensor[1][1])
			
			WeightTensorArrayArray[i] = Model:getWeightTensorArray(true)

		end

		local minimumMaximumCurrentQValue = math.min(table.unpack(maximumCurrentQValueArray))

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * minimumMaximumCurrentQValue)
		
		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, previousAction)
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}
		
		local temporalDifferenceErrorTensor = AqwamTensorLibrary:createTensor({1, 2})
		
		if (EligibilityTrace) then
			
			EligibilityTrace:increment(1, actionIndex, discountFactor, outputDimensionSizeArray)

		end

		for i = 1, 2, 1 do

			Model:setWeightTensorArray(WeightTensorArrayArray[i], true)

			local previousQTensor = Model:forwardPropagate(previousFeatureTensor, true)

			local previousQTensor = previousQTensor[1][actionIndex]
			
			local temporalDifferenceError = targetQValue - previousQTensor
			
			local lossTensor = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

			lossTensor[1][actionIndex] = temporalDifferenceError
			
			temporalDifferenceErrorTensor[1][i] = temporalDifferenceError
			
			if (EligibilityTrace) then lossTensor = EligibilityTrace:calculate(lossTensor) end
			
			local negatedLossTensor = AqwamTensorLibrary:unaryMinus(lossTensor) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error Tensor to make the neural network to perform gradient ascent.
			
			Model:update(negatedLossTensor, true)
			
			WeightTensorArrayArray[i] = Model:getWeightTensorArray(true)

		end
		
		return temporalDifferenceErrorTensor

	end)
	
	NewDeepClippedDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewDeepClippedDoubleQLearningModel.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepClippedDoubleQLearningModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepClippedDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepClippedDoubleQLearningModel

end

function DeepClippedDoubleQLearningModel:setWeightTensorArray1(WeightTensorArray1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[1] = WeightTensorArray1

	else

		self.WeightTensorArrayArray[1] = self:deepCopyTable(WeightTensorArray1)

	end

end

function DeepClippedDoubleQLearningModel:setWeightTensorArray2(WeightTensorArray2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.WeightTensorArrayArray[2] = WeightTensorArray2

	else

		self.WeightTensorArrayArray[2] = self:deepCopyTable(WeightTensorArray2)

	end

end

function DeepClippedDoubleQLearningModel:getWeightTensorArray1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[1]

	else

		return self:deepCopyTable(self.WeightTensorArrayArray[1])

	end

end

function DeepClippedDoubleQLearningModel:getWeightTensorArray2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.WeightTensorArrayArray[2]

	else

		return self:deepCopyTable(self.WeightTensorArrayArray[2])

	end

end

return DeepClippedDoubleQLearningModel
