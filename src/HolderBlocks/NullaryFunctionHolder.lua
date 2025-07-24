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

local BaseHolderBlock = require(script.Parent.BaseHolderBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

NullaryFunctionHolderBlock = {}

NullaryFunctionHolderBlock.__index = NullaryFunctionHolderBlock

setmetatable(NullaryFunctionHolderBlock, BaseHolderBlock)

function NullaryFunctionHolderBlock.new(parameterDictionary)

	local NewNullaryFunctionHolderBlock = BaseHolderBlock.new()

	setmetatable(NewNullaryFunctionHolderBlock, NullaryFunctionHolderBlock)

	NewNullaryFunctionHolderBlock:setName("NullaryFunctionHolder")
	
	NewNullaryFunctionHolderBlock:setRequiresInputTensors(false)
	
	NewNullaryFunctionHolderBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	NewNullaryFunctionHolderBlock:setFirstDerivativeFunctionRequiresTransformedTensor(true)
	
	local Function = parameterDictionary.Function
	
	local ChainRuleFirstDerivativeFunction = parameterDictionary.ChainRuleFirstDerivativeFunction
	
	local FirstDerivativeFunction = parameterDictionary.FirstDerivativeFunction
	
	if (not Function) then error("No nullary function.") end
	
	if (not ChainRuleFirstDerivativeFunction) then error("No chain rule first derivative nullary function.") end
	
	NewNullaryFunctionHolderBlock.Function = Function
	
	NewNullaryFunctionHolderBlock.ChainRuleFirstDerivativeFunction = ChainRuleFirstDerivativeFunction
	
	NewNullaryFunctionHolderBlock.FirstDerivativeFunction = FirstDerivativeFunction
	
	NewNullaryFunctionHolderBlock:setFunction(NewNullaryFunctionHolderBlock.Function)
	
	NewNullaryFunctionHolderBlock:setChainRuleFirstDerivativeFunction(NewNullaryFunctionHolderBlock.ChainRuleFirstDerivativeFunction)

	NewNullaryFunctionHolderBlock:setFirstDerivativeFunction(NewNullaryFunctionHolderBlock.FirstDerivativeFunction)

	return NewNullaryFunctionHolderBlock

end

return NullaryFunctionHolderBlock