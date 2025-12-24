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

local BaseExpansionBlock = {}

BaseExpansionBlock.__index = BaseExpansionBlock

setmetatable(BaseExpansionBlock, BaseFunctionBlock)

function BaseExpansionBlock.new()

	local NewBaseExpansionBlock = BaseFunctionBlock.new()

	setmetatable(NewBaseExpansionBlock, BaseExpansionBlock)

	NewBaseExpansionBlock:setName("BaseExpansionBlock")

	NewBaseExpansionBlock:setClassName("ExpansionBlock")

	NewBaseExpansionBlock:setSaveInputTensorArray(true)
	
	NewBaseExpansionBlock:setSaveTransformedTensor(false)

	NewBaseExpansionBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	NewBaseExpansionBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)

	return NewBaseExpansionBlock

end

return BaseExpansionBlock
