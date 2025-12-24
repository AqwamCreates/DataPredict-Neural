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

local GaussianBlock = {}

GaussianBlock.__index = GaussianBlock

setmetatable(GaussianBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	local functionToApply = function (z) return math.exp(-math.pow(z, 2)) end

	return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local functionToApply = function (z) return -2 * z * math.exp(-math.pow(z, 2)) end

	local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

	return {chainRuleFirstDerivativeTensor}

end

function GaussianBlock.new()

	local NewGaussianBlock = BaseActivationBlock.new()

	setmetatable(NewGaussianBlock, GaussianBlock)

	NewGaussianBlock:setName("Gaussian")

	NewGaussianBlock:setFunction(Function)

	NewGaussianBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewGaussianBlock

end

return GaussianBlock
