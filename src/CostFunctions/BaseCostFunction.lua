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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

BaseCostFunction = {}

BaseCostFunction.__index = BaseCostFunction

setmetatable(BaseCostFunction, BaseInstance)

function BaseCostFunction.new()

	local NewBaseCostFunction = BaseInstance.new()

	setmetatable(NewBaseCostFunction, BaseCostFunction)
	
	NewBaseCostFunction:setName("BaseCostFunction")
	
	NewBaseCostFunction:setClassName("CostFunction")

	return NewBaseCostFunction

end

function BaseCostFunction:setCostFunction(CostFunction)

	self.CostFunction = CostFunction

end

function BaseCostFunction:setLossFunction(LossFunction)

	self.LossFunction = LossFunction

end

function BaseCostFunction:calculateCostValue(generatedLabelTensor, labelTensor)

	local costValue = self.CostFunction(generatedLabelTensor, labelTensor)
	
	costValue = costValue / #generatedLabelTensor

	return costValue

end

function BaseCostFunction:calculateLossTensor(generatedLabelTensor, labelTensor)

	local lossTensor = self.LossFunction(generatedLabelTensor, labelTensor)

	return lossTensor

end

return BaseCostFunction