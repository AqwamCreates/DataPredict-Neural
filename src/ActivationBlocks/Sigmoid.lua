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

SigmoidBlock = {}

SigmoidBlock.__index = SigmoidBlock

setmetatable(SigmoidBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	local functionToApply = function(z) return 1/(1 + math.exp(-1 * z)) end

	return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local functionToApply = function (a) return (a * (1 - a)) end

	local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

	return {chainRuleFirstDerivativeTensor}

end

function SigmoidBlock.new()

	local NewSigmoidBlock = BaseActivationBlock.new()

	setmetatable(NewSigmoidBlock, SigmoidBlock)

	NewSigmoidBlock:setName("Sigmoid")

	NewSigmoidBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(true)

	NewSigmoidBlock:setFunction(Function)

	NewSigmoidBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewSigmoidBlock

end

return SigmoidBlock