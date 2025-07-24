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

local BaseAttentionBlock = {}

BaseAttentionBlock.__index = BaseAttentionBlock

setmetatable(BaseAttentionBlock, BaseFunctionBlock)

function BaseAttentionBlock.new()
	
	local NewBaseAttentionBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBaseAttentionBlock, BaseAttentionBlock)
	
	NewBaseAttentionBlock:setName("BaseAttentionBlock")
	
	NewBaseAttentionBlock:setClassName("AttentionBlock")
	
	NewBaseAttentionBlock:setSaveInputTensorArray(true)
	
	NewBaseAttentionBlock:setSaveTransformedTensor(true)

	NewBaseAttentionBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	return NewBaseAttentionBlock
	
end

return BaseAttentionBlock
