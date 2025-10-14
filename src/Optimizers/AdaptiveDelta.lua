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

local BaseOptimizer = require(script.Parent.BaseOptimizer)

AdaptiveDeltaOptimizer = {}

AdaptiveDeltaOptimizer.__index = AdaptiveDeltaOptimizer

setmetatable(AdaptiveDeltaOptimizer, BaseOptimizer)

local defaultDecayRate = 0.9

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function AdaptiveDeltaOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveDeltaOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveDeltaOptimizer, AdaptiveDeltaOptimizer)
	
	NewAdaptiveDeltaOptimizer:setName("AdaptiveDelta")
	
	NewAdaptiveDeltaOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	NewAdaptiveDeltaOptimizer.weightDecayRate = NewAdaptiveDeltaOptimizer.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveDeltaOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveDeltaOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewAdaptiveDeltaOptimizer.optimizerInternalParameterArray or {}
		
		local previousRunningGradientSquaredTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local decayRate = NewAdaptiveDeltaOptimizer.decayRate
		
		local weightDecayRate = NewAdaptiveDeltaOptimizer.weightDecayRate

		local gradientTensor = costFunctionDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local gradientSquaredTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local runningDeltaTensorPart1 = AqwamTensorLibrary:multiply(decayRate, previousRunningGradientSquaredTensor)

		local runningDeltaTensorPart2 = AqwamTensorLibrary:multiply((1 - decayRate), gradientSquaredTensor)

		local currentRunningGradientSquaredTensor = AqwamTensorLibrary:add(runningDeltaTensorPart1, runningDeltaTensorPart2)

		local rootMeanSquareTensorPart1 = AqwamTensorLibrary:add(currentRunningGradientSquaredTensor, NewAdaptiveDeltaOptimizer.epsilon)

		local rootMeanSquareTensor = AqwamTensorLibrary:applyFunction(math.sqrt, rootMeanSquareTensorPart1)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, rootMeanSquareTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)

		NewAdaptiveDeltaOptimizer.optimizerInternalParameterArray = {currentRunningGradientSquaredTensor}

		return costFunctionDerivativeTensor
		
	end)

	return NewAdaptiveDeltaOptimizer

end

function AdaptiveDeltaOptimizer:setDecayRate(decayRate)

	self.decayRate = decayRate

end

function AdaptiveDeltaOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveDeltaOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveDeltaOptimizer
