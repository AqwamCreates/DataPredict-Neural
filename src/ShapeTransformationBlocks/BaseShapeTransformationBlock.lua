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

BaseShapeTransformationBlock = {}

BaseShapeTransformationBlock.__index = BaseShapeTransformationBlock

setmetatable(BaseShapeTransformationBlock, BaseFunctionBlock)

function BaseShapeTransformationBlock.new()
	
	local NewBaseShapeTransformationBlock = BaseFunctionBlock.new()
	
	setmetatable(NewBaseShapeTransformationBlock, BaseShapeTransformationBlock)
	
	NewBaseShapeTransformationBlock:setName("BaseShapeTransformationBlock")
	
	NewBaseShapeTransformationBlock:setClassName("ShapeTransformationBlock")
	
	NewBaseShapeTransformationBlock:setSaveInputTensorArray(true)
	
	NewBaseShapeTransformationBlock:setSaveTransformedTensor(true)
	
	NewBaseShapeTransformationBlock:setSaveTotalFirstDerivativeTensorArray(true)
	
	NewBaseShapeTransformationBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	return NewBaseShapeTransformationBlock
	
end

return BaseShapeTransformationBlock