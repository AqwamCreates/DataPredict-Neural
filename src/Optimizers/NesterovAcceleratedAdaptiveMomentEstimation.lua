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

local NesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

NesterovAcceleratedAdaptiveMomentEstimationOptimizer.__index = NesterovAcceleratedAdaptiveMomentEstimationOptimizer

setmetatable(NesterovAcceleratedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer, NesterovAcceleratedAdaptiveMomentEstimationOptimizer)
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setName("NesterovAcceleratedAdaptiveMomentEstimation")
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray or {}
		
		local previousMTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local previousNTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1
		
		local beta1 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1
		
		local beta2 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2
		
		local weightDecayRate = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.weightDecayRate
		
		local gradientTensor = costFunctionDerivativeTensor
		
		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end
		
		local oneMinusBeta1 = (1 - beta1)

		local meanCostFunctionDerivativeTensor = AqwamTensorLibrary:divide(gradientTensor, oneMinusBeta1)

		local mTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMTensor)

		local mTensorPart2 = AqwamTensorLibrary:multiply(oneMinusBeta1, gradientTensor)

		local mTensor = AqwamTensorLibrary:add(mTensorPart1, mTensorPart2)

		local meanMTensor = AqwamTensorLibrary:divide(mTensor, oneMinusBeta1)

		local squaredGradientDerivativeTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local nTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousNTensor)

		local nTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeTensor)

		local nTensor = AqwamTensorLibrary:add(nTensorPart1, nTensorPart2)
		
		local multipliedNTensor = AqwamTensorLibrary:multiply(beta2, nTensor)

		local meanNTensor = AqwamTensorLibrary:divide(multipliedNTensor, (1 - math.pow(beta2, timeValue)))

		local finalMTensorPart1 = AqwamTensorLibrary:multiply(oneMinusBeta1, meanCostFunctionDerivativeTensor)

		local finalMTensorPart2 = AqwamTensorLibrary:multiply(beta1, meanMTensor)

		local finalMTensor = AqwamTensorLibrary:add(finalMTensorPart1, finalMTensorPart2)

		local squareRootedDivisor = AqwamTensorLibrary:applyFunction(math.sqrt, meanNTensor)

		local finalDivisor = AqwamTensorLibrary:add(squareRootedDivisor, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(finalMTensor, finalDivisor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)
		
		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {mTensor, nTensor, timeValue}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return NesterovAcceleratedAdaptiveMomentEstimationOptimizer
