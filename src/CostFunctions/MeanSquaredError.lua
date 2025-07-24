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

MeanSquaredError = {}

MeanSquaredError.__index = MeanSquaredError

setmetatable(MeanSquaredError, BaseCostFunction)

local function CostFunction(generatedLabelTensor, labelTensor)

	local functionToApply = function (generatedLabelValue, labelValue) return math.pow((generatedLabelValue - labelValue), 2) end
	
	local squaredErrorTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)
	
	local sumSquaredErrorValue = AqwamTensorLibrary:sum(squaredErrorTensor)

	local meanSquaredErrorValue = sumSquaredErrorValue / (#labelTensor)

	return meanSquaredErrorValue

end

local function LossFunction(generatedLabelTensor, labelTensor)
	
	local lossTensor = AqwamTensorLibrary:subtract(generatedLabelTensor, labelTensor)

	return AqwamTensorLibrary:multiply(2, lossTensor)

end

function MeanSquaredError.new()

	local NewMeanSquaredError = BaseCostFunction.new()

	setmetatable(NewMeanSquaredError, MeanSquaredError)

	NewMeanSquaredError:setName("MeanSquaredError")

	NewMeanSquaredError:setCostFunction(CostFunction)

	NewMeanSquaredError:setLossFunction(LossFunction)

	return NewMeanSquaredError

end

return MeanSquaredError
