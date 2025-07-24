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

local BaseFunctionBlock = require(script.Parent.Parent.Cores.BaseFunctionBlock)

local BaseHolderBlock = {}

BaseHolderBlock.__index = BaseHolderBlock

setmetatable(BaseHolderBlock, BaseFunctionBlock)

function BaseHolderBlock.new()
	
	local NewBaseNodeBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBaseNodeBlock, BaseHolderBlock)
	
	NewBaseNodeBlock:setName("BaseHolderBlock")
	
	NewBaseNodeBlock:setClassName("HolderBlock")
	
	NewBaseNodeBlock:setSaveTotalFirstDerivativeTensorArray(false)

	NewBaseNodeBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewBaseNodeBlock:setSaveInputTensorArray(true)
	
	NewBaseNodeBlock:setSaveTransformedTensor(false)

	NewBaseNodeBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	return NewBaseNodeBlock
	
end

return BaseHolderBlock
