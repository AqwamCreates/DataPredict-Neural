--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseGradientClipper = require(script.Parent.BaseGradientClipper)

ClipNormalizationGradientClipper = {}

ClipNormalizationGradientClipper.__index = ClipNormalizationGradientClipper

setmetatable(ClipNormalizationGradientClipper, BaseGradientClipper)

local defaultNormalizationValue = 2

function ClipNormalizationGradientClipper.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewClipNormalizationGradientClipper = BaseGradientClipper.new(parameterDictionary)
	
	setmetatable(NewClipNormalizationGradientClipper, ClipNormalizationGradientClipper)
	
	NewClipNormalizationGradientClipper:setName("ClipNormalization")
	
	local normalizationValue = parameterDictionary.normalizationValue or defaultNormalizationValue
	
	NewClipNormalizationGradientClipper.normalizationValue = normalizationValue
	
	NewClipNormalizationGradientClipper.maximumNormalizationValue = parameterDictionary.maximumNormalizationValue or normalizationValue
	
	--------------------------------------------------------------------------------
	
	NewClipNormalizationGradientClipper:setClipFunction(function(costFunctionDerivativeTensor)
		
		local normalizationValue = NewClipNormalizationGradientClipper.normalizationValue
		
		local maximumNormalizationValue = NewClipNormalizationGradientClipper.maximumNormalizationValue
		
		local squaredCostFunctionDerivativeTensor = AqwamTensorLibrary:power(costFunctionDerivativeTensor, normalizationValue)
		
		local sumSquaredCostFunctionDerivativeTensor = AqwamTensorLibrary:sum(squaredCostFunctionDerivativeTensor)
		
		local currentNormalizationValue = math.pow(sumSquaredCostFunctionDerivativeTensor, (1 / normalizationValue))
		
		if (currentNormalizationValue ~= 0) then
			
			costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(costFunctionDerivativeTensor, (maximumNormalizationValue / currentNormalizationValue))
		
		end
		
		return costFunctionDerivativeTensor
		
	end)
	
	return NewClipNormalizationGradientClipper
	
end

function ClipNormalizationGradientClipper:setNormalizationValue(normalizationValue)

	self.normalizationValue = normalizationValue

end

function ClipNormalizationGradientClipper:setMaximumNormalizationValue(maximumNormalizationValue)

	self.maximumNormalizationValue = maximumNormalizationValue

end

return ClipNormalizationGradientClipper
