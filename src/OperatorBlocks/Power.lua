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

local BaseOperatorBlock = require(script.Parent.BaseOperatorBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local PowerBlock = {}

PowerBlock.__index = PowerBlock

setmetatable(PowerBlock, BaseOperatorBlock)

function PowerBlock.new()

	local NewPowerBlock = BaseOperatorBlock.new()

	setmetatable(NewPowerBlock, PowerBlock)

	NewPowerBlock:setName("Power")
	
	NewPowerBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewPowerBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:power(table.unpack(inputTensorArray))
	
	end)

	NewPowerBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		local inputTensor1 = inputTensorArray[1]

		local inputTensor2 = inputTensorArray[2]

		local dimensionSizeArray1 = AqwamTensorLibrary:getDimensionSizeArray(inputTensor1)
		
		local dimensionSizeArray2 = AqwamTensorLibrary:getDimensionSizeArray(inputTensor2)
		
		local exponentMinusOneTensor = AqwamTensorLibrary:subtract(inputTensor2, 1)
		
		local chainRuleFirstDerivativeTensor1Part1 = AqwamTensorLibrary:power(inputTensor1, exponentMinusOneTensor)
		
		local chainRuleFirstDerivativeTensor1 = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, inputTensor1, chainRuleFirstDerivativeTensor1Part1)
		
		chainRuleFirstDerivativeTensorArray[1] = NewPowerBlock:collapseTensor(chainRuleFirstDerivativeTensor1, dimensionSizeArray1)
		
		local partialChainRuleFirstDerivativeTensor2 = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, inputTensor1, inputTensor2)
		
		local chainRuleFirstDerivativeTensor2 = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, partialChainRuleFirstDerivativeTensor2)
		
		chainRuleFirstDerivativeTensorArray[2] = NewPowerBlock:collapseTensor(chainRuleFirstDerivativeTensor2, dimensionSizeArray2)

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewPowerBlock

end

return PowerBlock
