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

AdaptiveMomentEstimationWeightDecayOptimizer = {}

AdaptiveMomentEstimationWeightDecayOptimizer.__index = AdaptiveMomentEstimationWeightDecayOptimizer

setmetatable(AdaptiveMomentEstimationWeightDecayOptimizer, BaseOptimizer)

local defaultAlpha = 0.001

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0.01

local defaultEpsilon = 1e-16

function AdaptiveMomentEstimationWeightDecayOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationWeightDecayOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer:setName("AdaptiveMomentEstimationWeightDecay")
	
	setmetatable(NewAdaptiveMomentEstimationWeightDecayOptimizer, AdaptiveMomentEstimationWeightDecayOptimizer)
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.alpha = parameterDictionary.alpha or defaultBeta1
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewAdaptiveMomentEstimationWeightDecayOptimizer.optimizerInternalParameterArray or {}
		
		local previousMomentumTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local previousVelocityTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1
		
		local beta1 = NewAdaptiveMomentEstimationWeightDecayOptimizer.beta1

		local beta2 = NewAdaptiveMomentEstimationWeightDecayOptimizer.beta2

		local decayedWeightTensor = AqwamTensorLibrary:multiply(NewAdaptiveMomentEstimationWeightDecayOptimizer.weightDecayRate, weightTensor)

		local gradientTensor = AqwamTensorLibrary:add(costFunctionDerivativeTensor, decayedWeightTensor)
		
		local momentumTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMomentumTensor)

		local momentumTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), gradientTensor)

		local momentumTensor = AqwamTensorLibrary:add(momentumTensorPart1, momentumTensorPart2)

		local squaredGradientDerivativeTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		local meanMomentumTensor = AqwamTensorLibrary:divide(momentumTensor, (1 - math.pow(beta1, timeValue)))

		local meanVelocityTensor = AqwamTensorLibrary:divide(velocityTensor, (1 - math.pow(beta2, timeValue)))

		local squareRootedDivisor = AqwamTensorLibrary:applyFunction(math.sqrt, meanVelocityTensor)

		local finalDivisorTensor = AqwamTensorLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationWeightDecayOptimizer.epsilon)
		
		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:multiply(NewAdaptiveMomentEstimationWeightDecayOptimizer.alpha, meanMomentumTensor)

		local costFunctionDerivativeTensorPart2 = AqwamTensorLibrary:divide(costFunctionDerivativeTensorPart1, finalDivisorTensor)
		
		local costFunctionDerivativeTensorPart3 = AqwamTensorLibrary:add(costFunctionDerivativeTensorPart2, decayedWeightTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart3)
		
		NewAdaptiveMomentEstimationWeightDecayOptimizer.optimizerInternalParameterArray = {momentumTensor, velocityTensor, timeValue}

		return costFunctionDerivativeTensor
		
	end)

	return NewAdaptiveMomentEstimationWeightDecayOptimizer

end

function AdaptiveMomentEstimationWeightDecayOptimizer:setAlpha(alpha)

	self.alpha = alpha

end

function AdaptiveMomentEstimationWeightDecayOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationWeightDecayOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationWeightDecayOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveMomentEstimationWeightDecayOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationWeightDecayOptimizer
