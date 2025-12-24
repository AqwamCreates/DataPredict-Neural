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

local BaseCostFunction = require(script.Parent.BaseCostFunction)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BinaryCrossEntropy = {}

BinaryCrossEntropy.__index = BinaryCrossEntropy

setmetatable(BinaryCrossEntropy, BaseCostFunction)

local function CostFunction(generatedLabelTensor, labelTensor)

	local functionToApply = function (generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue) + (1 - labelValue) * math.log(1 - generatedLabelValue)) end
	
	local binaryCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)
	
	local sumBinaryCrossEntropyValue = AqwamTensorLibrary:sum(binaryCrossEntropyTensor)
	
	local meanBinaryCrossEntropyValue = sumBinaryCrossEntropyValue / (#labelTensor)

	return meanBinaryCrossEntropyValue

end

local function LossFunction(generatedLabelTensor, labelTensor)
	
	local functionToApply = function (generatedLabelValue, labelValue) return ((generatedLabelValue - labelValue) / (generatedLabelValue * (1 - generatedLabelValue))) end

	return AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)

end

function BinaryCrossEntropy.new()

	local NewBinaryCrossEntropy = BaseCostFunction.new()

	setmetatable(NewBinaryCrossEntropy, BinaryCrossEntropy)

	NewBinaryCrossEntropy:setName("BinaryCrossEntropy")

	NewBinaryCrossEntropy:setCostFunction(CostFunction)

	NewBinaryCrossEntropy:setLossFunction(LossFunction)

	return NewBinaryCrossEntropy

end

return BinaryCrossEntropy
