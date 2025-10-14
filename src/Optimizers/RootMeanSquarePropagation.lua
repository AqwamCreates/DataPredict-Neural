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

RootMeanSquarePropagationOptimizer = {}

RootMeanSquarePropagationOptimizer.__index = RootMeanSquarePropagationOptimizer

setmetatable(RootMeanSquarePropagationOptimizer, BaseOptimizer)

local defaultBeta = 0.1

local defaultWeightDecayRate = 0

local defaultEpsilonValue = 1e-16

function RootMeanSquarePropagationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewRootMeanSquarePropagationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewRootMeanSquarePropagationOptimizer, RootMeanSquarePropagationOptimizer)
	
	NewRootMeanSquarePropagationOptimizer:setName("RootMeanSquarePropagation")
	
	NewRootMeanSquarePropagationOptimizer.beta = parameterDictionary.beta or defaultBeta
	
	NewRootMeanSquarePropagationOptimizer.weightDecayRate = NewRootMeanSquarePropagationOptimizer.weightDecayRate or defaultWeightDecayRate
	
	NewRootMeanSquarePropagationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilonValue
	
	--------------------------------------------------------------------------------
	
	NewRootMeanSquarePropagationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewRootMeanSquarePropagationOptimizer.optimizerInternalParameterArray or {}
		
		local previousVelocity = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local beta = NewRootMeanSquarePropagationOptimizer.beta
		
		local weightDecayRate = NewRootMeanSquarePropagationOptimizer.weightDecayRate
		
		local gradientTensor = costFunctionDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local squaredCostFunctionDerivativeTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local vTensorPart1 = AqwamTensorLibrary:multiply(beta, previousVelocity)

		local vTensorPart2 = AqwamTensorLibrary:multiply((1 - beta), squaredCostFunctionDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(vTensorPart1, vTensorPart2)

		local velocityNonZeroDivisorTensor = AqwamTensorLibrary:add(velocityTensor, NewRootMeanSquarePropagationOptimizer.epsilon)

		local squaredRootVelocityTensor = AqwamTensorLibrary:applyFunction(math.sqrt, velocityNonZeroDivisorTensor)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, squaredRootVelocityTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)

		NewRootMeanSquarePropagationOptimizer.optimizerInternalParameterArray = {velocityTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewRootMeanSquarePropagationOptimizer
	
end

function RootMeanSquarePropagationOptimizer:setBeta(beta)
	
	self.beta = beta
	
end

function RootMeanSquarePropagationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function RootMeanSquarePropagationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return RootMeanSquarePropagationOptimizer
