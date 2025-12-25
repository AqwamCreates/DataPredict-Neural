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

local AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

local defaultWeightDecayRate = 0

function AdaptiveGradientOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer:setName("AdaptiveGradient")
	
	NewAdaptiveGradientOptimizer.weightDecayRate = NewAdaptiveGradientOptimizer.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewAdaptiveGradientOptimizer.optimizerInternalParameterArray or {}
		
		local previousSumOfGradientSquaredTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local weightDecayRate = NewAdaptiveGradientOptimizer.weightDecayRate
		
		local gradientTensor = costFunctionDerivativeTensor
		
		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end
		
		local gradientSquaredTensor = AqwamTensorLibrary:power(gradientTensor, 2)

		local currentSumOfGradientSquaredTensor = AqwamTensorLibrary:add(previousSumOfGradientSquaredTensor, gradientSquaredTensor)

		local squareRootSumOfGradientSquaredTensor = AqwamTensorLibrary:applyFunction(math.sqrt, currentSumOfGradientSquaredTensor)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, squareRootSumOfGradientSquaredTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)

		NewAdaptiveGradientOptimizer.optimizerInternalParameterArray = {currentSumOfGradientSquaredTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

function AdaptiveGradientOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

return AdaptiveGradientOptimizer
