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

local BaseActivationBlock = {}

BaseActivationBlock.__index = BaseActivationBlock

setmetatable(BaseActivationBlock, BaseFunctionBlock)

function BaseActivationBlock.new()
	
	local NewBaseActivationBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBaseActivationBlock, BaseActivationBlock)
	
	NewBaseActivationBlock:setName("BaseDropoutBlock")
	
	NewBaseActivationBlock:setClassName("DropoutBlock")
	
	NewBaseActivationBlock:setSaveInputTensorArray(true)
	
	NewBaseActivationBlock:setSaveTransformedTensor(true)

	NewBaseActivationBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	return NewBaseActivationBlock
	
end

return BaseActivationBlock
