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

TanhBlock = {}

TanhBlock.__index = TanhBlock

setmetatable(TanhBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	return AqwamTensorLibrary:applyFunction(math.tanh, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local functionToApply = function (a) return (1 - math.pow(a, 2)) end

	local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

	return {chainRuleFirstDerivativeTensor}

end

function TanhBlock.new()

	local NewTanhBlock = BaseActivationBlock.new()

	setmetatable(NewTanhBlock, TanhBlock)

	NewTanhBlock:setName("Tanh")

	NewTanhBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(true)

	NewTanhBlock:setFunction(Function)

	NewTanhBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewTanhBlock

end

return TanhBlock