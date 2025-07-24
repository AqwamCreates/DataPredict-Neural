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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local BaseModel = {}

BaseModel.__index = BaseModel

setmetatable(BaseModel, BaseInstance)

function BaseModel.new()
	
	local NewBaseModel = BaseInstance.new()
	
	setmetatable(NewBaseModel, BaseModel)
	
	NewBaseModel:setName("BaseModel")
	
	NewBaseModel:setClassName("Model")
	
	return NewBaseModel
	
end

return BaseModel
