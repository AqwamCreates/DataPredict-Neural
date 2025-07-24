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

LeakyRectifiedLinearUnitBlock = {}

LeakyRectifiedLinearUnitBlock.__index = LeakyRectifiedLinearUnitBlock

setmetatable(LeakyRectifiedLinearUnitBlock, BaseActivationBlock)

local defaultNegativeSlopeFactor = 0.01

function LeakyRectifiedLinearUnitBlock.new(parameterDictionary)

	local NewLeakyRectifiedLinearUnitBlock = BaseActivationBlock.new()

	setmetatable(NewLeakyRectifiedLinearUnitBlock, LeakyRectifiedLinearUnitBlock)

	NewLeakyRectifiedLinearUnitBlock:setName("LeakyRectifiedLinearUnit")

	parameterDictionary = parameterDictionary or {}

	NewLeakyRectifiedLinearUnitBlock.negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or defaultNegativeSlopeFactor

	NewLeakyRectifiedLinearUnitBlock:setFunction(function(inputTensorArray)

		local negativeSlopeFactor = NewLeakyRectifiedLinearUnitBlock.negativeSlopeFactor

		local functionToApply = function (z) return math.max(z, z * negativeSlopeFactor) end

		return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	end)

	NewLeakyRectifiedLinearUnitBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local negativeSlopeFactor = NewLeakyRectifiedLinearUnitBlock.negativeSlopeFactor

		local functionToApply = function (z) if (z >= 0) then return 1 else return negativeSlopeFactor end end

		local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewLeakyRectifiedLinearUnitBlock

end

function LeakyRectifiedLinearUnitBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.negativeSlopeFactor = parameterDictionary.negativeSlopeFactor or self.negativeSlopeFactor

end

return LeakyRectifiedLinearUnitBlock