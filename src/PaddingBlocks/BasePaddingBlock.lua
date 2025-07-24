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

local BasePaddingBlock = {}

BasePaddingBlock.__index = BasePaddingBlock

setmetatable(BasePaddingBlock, BaseFunctionBlock)

function BasePaddingBlock.new()
	
	local NewBasePaddingBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBasePaddingBlock, BasePaddingBlock)
	
	NewBasePaddingBlock:setName("BasePaddingBlock")
	
	NewBasePaddingBlock:setClassName("PaddingBlock")
	
	NewBasePaddingBlock:setSaveInputTensorArray(true)
	
	NewBasePaddingBlock:setSaveTransformedTensor(true)

	NewBasePaddingBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	return NewBasePaddingBlock
	
end

return BasePaddingBlock
