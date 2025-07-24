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

MeanAbsoluteError = {}

MeanAbsoluteError.__index = MeanAbsoluteError

setmetatable(MeanAbsoluteError, BaseCostFunction)

local function CostFunction(generatedLabelTensor, labelTensor)

	local functionToApply = function (generatedLabelValue, labelValue) return math.abs(generatedLabelValue - labelValue) end
	
	local absoluteErrorTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)
	
	local sumAbsoluteErrorValue = AqwamTensorLibrary:sum(absoluteErrorTensor)
	
	local meanAbsoluteErrorValue = sumAbsoluteErrorValue / (#labelTensor)

	return meanAbsoluteErrorValue

end

local function LossFunction(generatedLabelTensor, labelTensor)

	return AqwamTensorLibrary:subtract(generatedLabelTensor, labelTensor)

end

function MeanAbsoluteError.new()

	local NewMeanAbsoluteError = BaseCostFunction.new()

	setmetatable(NewMeanAbsoluteError, MeanAbsoluteError)

	NewMeanAbsoluteError:setName("MeanAbsoluteError")

	NewMeanAbsoluteError:setCostFunction(CostFunction)

	NewMeanAbsoluteError:setLossFunction(LossFunction)

	return NewMeanAbsoluteError

end

return MeanAbsoluteError
