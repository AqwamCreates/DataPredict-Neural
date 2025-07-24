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

local BasePoolingBlock = {}

BasePoolingBlock.__index = BasePoolingBlock

setmetatable(BasePoolingBlock, BaseFunctionBlock)

function BasePoolingBlock.new()
	
	local NewBasePoolingBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBasePoolingBlock, BasePoolingBlock)
	
	NewBasePoolingBlock:setName("BasePoolingBlock")
	
	NewBasePoolingBlock:setClassName("PoolingBlock")
	
	NewBasePoolingBlock:setSaveInputTensorArray(true)
	
	NewBasePoolingBlock:setSaveTransformedTensor(true)

	NewBasePoolingBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	return NewBasePoolingBlock
	
end

return BasePoolingBlock
