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

local GravityOptimizer = {}

GravityOptimizer.__index = GravityOptimizer

setmetatable(GravityOptimizer, BaseOptimizer)

local defaultInitialStepSize = 0.01

local defaultMovingAverage = 0.9

local defaultWeightDecayRate = 0

local function calculateGaussianDensity(mean, standardDeviation)

	local exponentStep1 = math.pow(mean, 2)

	local exponentPart2 = math.pow(standardDeviation, 2)

	local exponentStep3 = exponentStep1 / exponentPart2

	local exponentStep4 = -0.5 * exponentStep3

	local exponentWithTerms = math.exp(exponentStep4)

	local divisor = standardDeviation * math.sqrt(2 * math.pi)

	local gaussianDensity = exponentWithTerms / divisor

	return gaussianDensity

end

function GravityOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGravityOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewGravityOptimizer, GravityOptimizer)
	
	NewGravityOptimizer:setName("Gravity")
	
	NewGravityOptimizer.initialStepSize = parameterDictionary.initialStepSize or defaultInitialStepSize
	
	NewGravityOptimizer.movingAverage = parameterDictionary.movingAverage or defaultMovingAverage
	
	NewGravityOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewGravityOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewGravityOptimizer.optimizerInternalParameterArray or {}
		
		local previousVelocityTensor = optimizerInternalParameterArray[1]
		
		local timeValue = (optimizerInternalParameterArray[2] or 0) + 1
		
		local weightDecayRate = NewGravityOptimizer.weightDecayRate
		
		if (not previousVelocityTensor) then

			local standardDeviation = NewGravityOptimizer.initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			previousVelocityTensor = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), gaussianDensity)

		end
		
		local gradientTensor = costFunctionDerivativeTensor
		
		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end

		local meanMovingAverage = ((NewGravityOptimizer.movingAverage * timeValue) + 1) / (timeValue + 2)

		local absoluteGradientTensor = AqwamTensorLibrary:applyFunction(math.abs, gradientTensor)

		local maximumGradientValue = AqwamTensorLibrary:findMaximumValue(absoluteGradientTensor)

		local mTensor = AqwamTensorLibrary:divide(1, maximumGradientValue)

		local weirdLTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, mTensor)

		local weirdLTensorPart2 = AqwamTensorLibrary:power(weirdLTensorPart1, 2)

		local weirdLTensorPart3 = AqwamTensorLibrary:add(1, weirdLTensorPart2)

		local weirdLTensor = AqwamTensorLibrary:divide(gradientTensor, weirdLTensorPart3)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(meanMovingAverage, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - meanMovingAverage), weirdLTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, velocityTensor)

		NewGravityOptimizer.optimizerInternalParameterArray = {velocityTensor, timeValue}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewGravityOptimizer
	
end

function GravityOptimizer:setInitialStepSize(initialStepSize)
	
	self.initialStepSize = initialStepSize
	
end

function GravityOptimizer:setMovingAverage(movingAverage)

	self.movingAverage = movingAverage

end

function GravityOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

return GravityOptimizer
