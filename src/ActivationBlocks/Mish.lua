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

MishBlock = {}

MishBlock.__index = MishBlock

setmetatable(MishBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	local functionToApply = function (z) return (z * math.tanh(math.log(1 + math.exp(z)))) end

	return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local functionToApply = function (z) 

		return (math.exp(z) * (math.exp(3 * z) + 4 * math.exp(2 * z) + (6 + 4 * z) * math.exp(z) + 4 * (1 + z)) / math.pow((1 + math.pow((math.exp(z) + 1), 2)), 2))

	end

	local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

	return {chainRuleFirstDerivativeTensor}

end

function MishBlock.new()

	local NewMishBlock = BaseActivationBlock.new()

	setmetatable(NewMishBlock, MishBlock)

	NewMishBlock:setName("Mish")

	NewMishBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(true)

	NewMishBlock:setFunction(Function)

	NewMishBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewMishBlock

end

return MishBlock