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

RectifiedAdaptiveMomentEstimationOptimizer = {}

RectifiedAdaptiveMomentEstimationOptimizer.__index = RectifiedAdaptiveMomentEstimationOptimizer

setmetatable(RectifiedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function RectifiedAdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRectifiedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewRectifiedAdaptiveMomentEstimationOptimizer:setName("RectifiedAdaptiveMomentEstimation")
	
	setmetatable(NewRectifiedAdaptiveMomentEstimationOptimizer, RectifiedAdaptiveMomentEstimationOptimizer)
	
	local beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.beta2 = beta2
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.pInfinity = ((2 / (1 - beta2)) - 1)
	
	--------------------------------------------------------------------------------
	
	NewRectifiedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local previousMomentumTensor = NewRectifiedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local previousVelocityTensor = NewRectifiedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local timeValue = NewRectifiedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[3] or 1
		
		local beta1 = NewRectifiedAdaptiveMomentEstimationOptimizer.beta1

		local beta2 = NewRectifiedAdaptiveMomentEstimationOptimizer.beta2
		
		local weightDecayRate = NewRectifiedAdaptiveMomentEstimationOptimizer.weightDecayRate
		
		local pInfinity = NewRectifiedAdaptiveMomentEstimationOptimizer.pInfinity

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
		
		local powerBeta2 = math.pow(beta2, timeValue)
		
		local p = pInfinity - ((2 * timeValue * powerBeta2) / (1 - powerBeta2))
		
		if (p > 4) then
			
			local squareRootVelocityTensor = AqwamTensorLibrary:applyFunction(math.sqrt, velocityTensor)
			
			local adaptiveLearningRateTensorPart1 = AqwamTensorLibrary:add(squareRootVelocityTensor, NewRectifiedAdaptiveMomentEstimationOptimizer.epsilon)
			
			local adaptiveLearningRateTensor = AqwamTensorLibrary:divide((1 - powerBeta2), adaptiveLearningRateTensorPart1)
			
			local varianceRectificationNominatorValue = (p - 4) * (p - 2) * pInfinity
			
			local varianceRectificationDenominatorValue = (pInfinity - 4) * (pInfinity - 2) * p
			
			local varianceRectificationValue =  math.sqrt(varianceRectificationNominatorValue / varianceRectificationDenominatorValue)
			
			costFunctionDerivativeTensor = AqwamTensorLibrary:multiply((learningRate * varianceRectificationValue), meanMomentumTensor, adaptiveLearningRateTensor)
			
		else
			
			costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, meanMomentumTensor)
			
		end
		
		timeValue = timeValue + 1
		
		NewRectifiedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {momentumTensor, velocityTensor, timeValue}

		return costFunctionDerivativeTensor
		
	end)

	return NewRectifiedAdaptiveMomentEstimationOptimizer

end

function RectifiedAdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function RectifiedAdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function RectifiedAdaptiveMomentEstimationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function RectifiedAdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return RectifiedAdaptiveMomentEstimationOptimizer
