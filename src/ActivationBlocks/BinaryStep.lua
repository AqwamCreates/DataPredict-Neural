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

BinaryStepBlock = {}

BinaryStepBlock.__index = BinaryStepBlock

setmetatable(BinaryStepBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	local functionToApply = function (z) return ((z > 0) and 1) or 0 end

	return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor), 0)

	return {chainRuleFirstDerivativeTensor}

end

function BinaryStepBlock.new()

	local NewBinaryStepBlock = BaseActivationBlock.new()

	setmetatable(NewBinaryStepBlock, BinaryStepBlock)

	NewBinaryStepBlock:setName("BinaryStep")

	NewBinaryStepBlock:setFunction(Function)

	NewBinaryStepBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewBinaryStepBlock

end

return BinaryStepBlock