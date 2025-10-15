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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

BaseEligibilityTrace = {}

BaseEligibilityTrace.__index = BaseEligibilityTrace

setmetatable(BaseEligibilityTrace, BaseInstance)

local defaultLambda = 0.5

function BaseEligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseEligibilityTrace = BaseInstance.new()
	
	setmetatable(NewBaseEligibilityTrace, BaseEligibilityTrace)
	
	NewBaseEligibilityTrace:setName("BaseEligibilityTrace")
	
	NewBaseEligibilityTrace:setClassName("EligibilityTrace")
	
	NewBaseEligibilityTrace.lambda = parameterDictionary.lambda or defaultLambda
	
	NewBaseEligibilityTrace.eligibilityTraceTensor = nil
	
	return NewBaseEligibilityTrace
	
end

function BaseEligibilityTrace:increment(actionIndex, discountFactor, dimensionSizeArray) -- This function is needed because we have double version of reinforcement learning algorithms require separate application of (temporalDifferenceErrorVector * eligibilityTraceMatrix).
	
	local eligibilityTraceTensor = self.eligibilityTraceTensor or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0) 

	eligibilityTraceTensor = AqwamTensorLibrary:multiply(eligibilityTraceTensor, discountFactor * self.lambda)
	
	local IncrementFunction = self.IncrementFunction
	
	if (IncrementFunction) then self.eligibilityTraceTensor = IncrementFunction(eligibilityTraceTensor, actionIndex) end
	
end

function BaseEligibilityTrace:calculate(temporalDifferenceError, actionIndex)
	
	local eligibilityTraceTensor = self.eligibilityTraceTensor
	
	if (not actionIndex) then
		
		return AqwamTensorLibrary:multiply(temporalDifferenceError, eligibilityTraceTensor)
		
	else
		
		return (temporalDifferenceError * eligibilityTraceTensor)
		
	end
	
end

function BaseEligibilityTrace:setIncrementFunction(IncrementFunction)
	
	self.IncrementFunction = IncrementFunction
	
end

function BaseEligibilityTrace:getLambda()
	
	return self.lambda
	
end

function BaseEligibilityTrace:setLambda(lambda)

	self.lambda = lambda

end

function BaseEligibilityTrace:reset()
	
	self.eligibilityTraceTensor = nil
	
end

return BaseEligibilityTrace
