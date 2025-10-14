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

MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

setmetatable(MomentumOptimizer, BaseOptimizer)

local defaultDecayRate = 0.1

local defaultWeightDecayRate = 0

function MomentumOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMomentumOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer:setName("Momentum")
	
	NewMomentumOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	NewMomentumOptimizer.weightDecayRate = NewMomentumOptimizer.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local optimizerInternalParameterArray = NewMomentumOptimizer.optimizerInternalParameterArray or {}
		
		local previousVelocityTensor = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local weightDecayRate = NewMomentumOptimizer.weightDecayRate

		local gradientTensor = costFunctionDerivativeTensor

		if (weightDecayRate ~= 0) then

			local decayedWeightTensor = AqwamTensorLibrary:multiply(weightDecayRate, weightTensor)

			gradientTensor = AqwamTensorLibrary:add(gradientTensor, decayedWeightTensor)

		end
		
		local velocityTensorPart1 = AqwamTensorLibrary:multiply(NewMomentumOptimizer.decayRate, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply(learningRate, gradientTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		costFunctionDerivativeTensor = velocityTensor
		
		NewMomentumOptimizer.optimizerInternalParameterArray = {velocityTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

function MomentumOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

return MomentumOptimizer
