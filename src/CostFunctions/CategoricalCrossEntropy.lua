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

CategoricalCrossEntropy = {}

CategoricalCrossEntropy.__index = CategoricalCrossEntropy

setmetatable(CategoricalCrossEntropy, BaseCostFunction)

local function CostFunction(generatedLabelTensor, labelTensor)

	local functionToApply = function (generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue)) end
	
	local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)
	
	local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

	local meanCategoricalCrossEntropyValue = sumCategoricalCrossEntropyValue / (#labelTensor)

	return meanCategoricalCrossEntropyValue

end

local function LossFunction(generatedLabelTensor, labelTensor)

	return AqwamTensorLibrary:subtract(generatedLabelTensor, labelTensor)

end

function CategoricalCrossEntropy.new()

	local NewCategoricalCrossEntropy = BaseCostFunction.new()

	setmetatable(NewCategoricalCrossEntropy, CategoricalCrossEntropy)

	NewCategoricalCrossEntropy:setName("CategoricalCrossEntropy")

	NewCategoricalCrossEntropy:setCostFunction(CostFunction)

	NewCategoricalCrossEntropy:setLossFunction(LossFunction)

	return NewCategoricalCrossEntropy

end

return CategoricalCrossEntropy
