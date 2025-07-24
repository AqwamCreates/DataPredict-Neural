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

FocalLoss = {}

FocalLoss.__index = FocalLoss

setmetatable(FocalLoss, BaseCostFunction)

local defaultAlpha = 0.25

local defaultGamma = 2

local function LossFunction(generatedLabelTensor, labelTensor)

	return AqwamTensorLibrary:subtract(generatedLabelTensor, labelTensor)

end

function FocalLoss.new(parameterDictionary)

	local NewFocalLoss = BaseCostFunction.new()

	setmetatable(NewFocalLoss, FocalLoss)
	
	parameterDictionary = parameterDictionary or {}
	
	NewFocalLoss.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewFocalLoss.gamma = parameterDictionary.gamma or defaultGamma

	NewFocalLoss:setName("FocalLoss")

	NewFocalLoss:setCostFunction(function(generatedLabelTensor, labelTensor)
		
		local alpha = NewFocalLoss.alpha 
		
		local gamma = NewFocalLoss.gamma 
		
		local functionToApply = function (generatedLabelValue, labelValue) 

			local isLabelValueEqualTo1 = (labelValue == 1)

			local pT = (isLabelValueEqualTo1 and generatedLabelValue) or (1 - generatedLabelValue)

			local focalLossValue = -alpha * ((1 - pT) ^ gamma) * math.log(pT)

			return focalLossValue

		end
		
		local focalLossTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)
		
		local sumFocalLossValue = AqwamTensorLibrary:sum(focalLossTensor)
		
		local meanFocalLossValue = sumFocalLossValue / (#labelTensor)
		
		return meanFocalLossValue
		
	end)

	NewFocalLoss:setLossFunction(function(generatedLabelTensor, labelTensor)
		
		local alpha = NewFocalLoss.alpha 

		local gamma = NewFocalLoss.gamma 

		local functionToApply = function (predictedValue, labelValue) 

			local isLabelValueEqualTo1 = (labelValue == 1)

			local pT = (isLabelValueEqualTo1 and predictedValue) or (1 - predictedValue)

			local focalLossValue = -alpha * ((1 - pT) ^ gamma) * ((gamma * pT * math.log(pT)) + pT - 1)

			return focalLossValue

		end
		
		local focalLossTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelTensor, labelTensor)
		
		return focalLossTensor
		
	end)

	return NewFocalLoss

end

function FocalLoss:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.alpha = parameterDictionary.alpha or self.alpha

	self.gamma = parameterDictionary.gamma or self.gamma
	
end

return FocalLoss
