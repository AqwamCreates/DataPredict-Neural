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

ResilientBackwardPropagationOptimizer = {}

ResilientBackwardPropagationOptimizer.__index = ResilientBackwardPropagationOptimizer

setmetatable(ResilientBackwardPropagationOptimizer, BaseOptimizer)

local defaultEtaPlus = 0.5

local defaultEtaMinus = 1.2

local defaultMinimumStepSize = 1e-6

local defaultMaximumStepSize = 50

local defaultWeightDecayRate = 0

function ResilientBackwardPropagationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewResilientBackwardPropagationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewResilientBackwardPropagationOptimizer, ResilientBackwardPropagationOptimizer)
	
	NewResilientBackwardPropagationOptimizer:setName("ResilientBackwardPropagation")
	
	NewResilientBackwardPropagationOptimizer.etaPlus = parameterDictionary.etaPlus or defaultEtaPlus
	
	NewResilientBackwardPropagationOptimizer.etaMinus = parameterDictionary.etaMinus or defaultEtaMinus
	
	NewResilientBackwardPropagationOptimizer.maximumStepSize = parameterDictionary.maximumStepSize or defaultMaximumStepSize
	
	NewResilientBackwardPropagationOptimizer.minimumStepSize = parameterDictionary.minimumStepSize or defaultMinimumStepSize
	
	NewResilientBackwardPropagationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewResilientBackwardPropagationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewResilientBackwardPropagationOptimizer.optimizerInternalParameterArray or {}
		
		local previousGradientTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local learningRateTensor = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), learningRate)
		
		local etaPlus = NewResilientBackwardPropagationOptimizer.etaPlus
		
		local etaMinus = NewResilientBackwardPropagationOptimizer.etaMinus
		
		local maximumStepSize = NewResilientBackwardPropagationOptimizer.maximumStepSize
		
		local minimumStepSize = NewResilientBackwardPropagationOptimizer.minimumStepSize
		
		local weightDecayRate = NewResilientBackwardPropagationOptimizer.weightDecayRate
		
		local gradientTensor = costFunctionDerivativeTensor
		
		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end
		
		local multipliedGradientTensor = AqwamTensorLibrary:multiply(gradientTensor, previousGradientTensor)
		
		for i, subMultipliedGradientTensor in ipairs(multipliedGradientTensor) do

			for j, gradientValue in ipairs(subMultipliedGradientTensor) do

				if (gradientValue > 0) then

					learningRateTensor[i][j] = math.min(learningRateTensor[i][j] * etaPlus, maximumStepSize)

				elseif (gradientValue < 0) then

					learningRateTensor[i][j] = math.max(learningRateTensor[i][j] * etaMinus, minimumStepSize)

					gradientTensor[i][j] = 0

				end

			end

		end
		
		local signTensor = AqwamTensorLibrary:applyFunction(math.sign, gradientTensor)
		
		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRateTensor, signTensor)

		NewResilientBackwardPropagationOptimizer.optimizerInternalParameterArray = {gradientTensor, learningRateTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewResilientBackwardPropagationOptimizer
	
end

function ResilientBackwardPropagationOptimizer:setEtaPlus(etaPlus)
	
	self.etaPlus = etaPlus
	
end

function ResilientBackwardPropagationOptimizer:setEtaMinus(etaMinus)

	self.etaMinus = etaMinus

end

function ResilientBackwardPropagationOptimizer:setMaximumStepSize(maximumStepSize)

	self.maximumStepSize = maximumStepSize

end

function ResilientBackwardPropagationOptimizer:setMinimumStepSize(minimumStepSize)

	self.minimumStepSize = minimumStepSize

end

return ResilientBackwardPropagationOptimizer
