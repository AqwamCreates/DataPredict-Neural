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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseFunctionBlock = require(script.Parent.Parent.Cores.BaseFunctionBlock)

BaseEncodingBlock = {}

BaseEncodingBlock.__index = BaseEncodingBlock

setmetatable(BaseEncodingBlock, BaseFunctionBlock)

function BaseEncodingBlock.new()
	
	local NewBaseEncodingBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBaseEncodingBlock, BaseEncodingBlock)
	
	NewBaseEncodingBlock:setName("BaseEncodingBlock")
	
	NewBaseEncodingBlock:setClassName("EncodingBlock")
	
	NewBaseEncodingBlock:setSaveInputTensorArray(true)
	
	NewBaseEncodingBlock:setSaveTransformedTensor(true)
	
	NewBaseEncodingBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	return NewBaseEncodingBlock
	
end

return BaseEncodingBlock
