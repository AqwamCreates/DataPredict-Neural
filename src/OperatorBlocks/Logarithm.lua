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

LogarithmBlock = {}

LogarithmBlock.__index = LogarithmBlock

setmetatable(LogarithmBlock, BaseOperatorBlock)

function LogarithmBlock.new()

	local NewLogarithmBlock = BaseOperatorBlock.new()

	setmetatable(NewLogarithmBlock, LogarithmBlock)

	NewLogarithmBlock:setName("Logarithm")
	
	NewLogarithmBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewLogarithmBlock:setFunction(function(inputTensorArray)
		
		return AqwamTensorLibrary:logarithm(table.unpack(inputTensorArray))
	
	end)

	NewLogarithmBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		local inputTensor1 = inputTensorArray[1]
		
		local inputTensor2 = inputTensorArray[2]
		
		if (inputTensor2) then
			
			local dimensionSizeArray1 = AqwamTensorLibrary:getDimensionSizeArray(inputTensor1)
			
			local collapsedDerivativeTensor1 = NewLogarithmBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray1)
			
			local partialDerivativeFunctionToApply1 = function (number, base) return (1 / (number * math.log(base))) end

			local partialDerivativeTensor1 = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply1, inputTensor1, inputTensor2)
			
			chainRuleFirstDerivativeTensorArray[1] = AqwamTensorLibrary:multiply(partialDerivativeTensor1, collapsedDerivativeTensor1)
			
			local dimensionSizeArray2 = AqwamTensorLibrary:getDimensionSizeArray(inputTensor2)
			
			local collapsedDerivativeTensor2 = NewLogarithmBlock:collapseTensor(initialPartialFirstDerivativeTensor, dimensionSizeArray2)

			local partialDerivativeFunctionToApply2 = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

			local partialDerivativeTensor2 = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply2, inputTensor1, inputTensor2)

			chainRuleFirstDerivativeTensorArray[2] = AqwamTensorLibrary:multiply(partialDerivativeTensor2, collapsedDerivativeTensor2)
			
		else
			
			local partialDerivativeFunctionToApply = function (number) return (1 / number) end

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, inputTensor1)
			
			chainRuleFirstDerivativeTensorArray[1] = AqwamTensorLibrary:multiply(partialDerivativeTensor, initialPartialFirstDerivativeTensor)
			
		end

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewLogarithmBlock

end

return LogarithmBlock