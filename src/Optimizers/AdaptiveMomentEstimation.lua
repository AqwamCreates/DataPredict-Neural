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

AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

setmetatable(AdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function AdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewAdaptiveMomentEstimationOptimizer:setName("AdaptiveMomentEstimation")
	
	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)
	
	NewAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local previousMomentumTensor = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local previousVelocityTensor = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local timeValue = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[3] or 1
		
		local beta1 = NewAdaptiveMomentEstimationOptimizer.beta1

		local beta2 = NewAdaptiveMomentEstimationOptimizer.beta2
		
		local weightDecayRate = NewAdaptiveMomentEstimationOptimizer.weightDecayRate

		local gradientTensor = costFunctionDerivativeTensor
		
		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end
		
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

		local finalDivisorTensor = AqwamTensorLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(meanMomentumTensor, finalDivisorTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)
		
		timeValue = timeValue + 1
		
		NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {momentumTensor, velocityTensor, timeValue}

		return costFunctionDerivativeTensor
		
	end)

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationOptimizer
