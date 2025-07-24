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

RectifiedLinearUnitBlock = {}

RectifiedLinearUnitBlock.__index = RectifiedLinearUnitBlock

setmetatable(RectifiedLinearUnitBlock, BaseActivationBlock)

local function Function(inputTensorArray)

	local functionToApply = function (z) return math.max(z, 0) end

	return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

end

local function ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local functionToApply = function (z) if (z >= 0) then return 1 else return 0 end end

	local gradientTensor = AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])

	local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, gradientTensor)

	return {chainRuleFirstDerivativeTensor}

end

function RectifiedLinearUnitBlock.new()

	local NewRectifiedLinearUnitBlock = BaseActivationBlock.new()

	setmetatable(NewRectifiedLinearUnitBlock, RectifiedLinearUnitBlock)

	NewRectifiedLinearUnitBlock:setName("RectifiedLinearUnit")

	NewRectifiedLinearUnitBlock:setFunction(Function)

	NewRectifiedLinearUnitBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	return NewRectifiedLinearUnitBlock

end

return RectifiedLinearUnitBlock