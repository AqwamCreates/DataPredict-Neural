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

SigmoidLinearUnitBlock = {}

SigmoidLinearUnitBlock.__index = SigmoidLinearUnitBlock

setmetatable(SigmoidLinearUnitBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	local functionToApply = function (z) return z / (1 + math.exp(-z)) end

	return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local functionToApply = function (z) return (1 + math.exp(-z) + (z * math.exp(-z))) / (1 + math.exp(-z))^2 end

	local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

	return {chainRuleFirstDerivativeTensor}

end

function SigmoidLinearUnitBlock.new()

	local NewSigmoidLinearUnitBlock = BaseActivationBlock.new()

	setmetatable(NewSigmoidLinearUnitBlock, SigmoidLinearUnitBlock)

	NewSigmoidLinearUnitBlock:setName("SigmoidLinearUnit")

	NewSigmoidLinearUnitBlock:setFunction(Function)

	NewSigmoidLinearUnitBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewSigmoidLinearUnitBlock

end

return SigmoidLinearUnitBlock