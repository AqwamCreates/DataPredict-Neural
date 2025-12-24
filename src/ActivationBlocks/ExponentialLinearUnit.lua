--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Neural)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseActivationBlock = require(script.Parent.BaseActivationBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ExponentialLinearUnitBlock = {}

ExponentialLinearUnitBlock.__index = ExponentialLinearUnitBlock

setmetatable(ExponentialLinearUnitBlock, BaseActivationBlock)

local defaultNegativeSlopeFactor = 0.01

function ExponentialLinearUnitBlock.new(parameterDictionary)

	local NewExponentialLinearUnitBlock = BaseActivationBlock.new()

	setmetatable(NewExponentialLinearUnitBlock, ExponentialLinearUnitBlock)

	NewExponentialLinearUnitBlock:setName("ExponentialLinearUnit")

	parameterDictionary = parameterDictionary or {}

	NewExponentialLinearUnitBlock.negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or defaultNegativeSlopeFactor

	NewExponentialLinearUnitBlock:setFunction(function(inputTensorArray)

		local negativeSlopeFactor = NewExponentialLinearUnitBlock.negativeSlopeFactor

		local functionToApply = function (z) return if (z > 0) then z else (negativeSlopeFactor * (math.exp(z) - 1)) end

		return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	end)

	NewExponentialLinearUnitBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local negativeSlopeFactor = NewExponentialLinearUnitBlock.negativeSlopeFactor

		local functionToApply =  function (z) if (z > 0) then return 1 else return (negativeSlopeFactor * math.exp(z)) end end

		local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewExponentialLinearUnitBlock

end

function ExponentialLinearUnitBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or self.negativeSlopeFactor

end

return ExponentialLinearUnitBlock
