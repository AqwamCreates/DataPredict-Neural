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

local BaseDropoutBlock = require(script.Parent.BaseDropoutBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local DropoutBlock = {}

DropoutBlock.__index = DropoutBlock

setmetatable(DropoutBlock, BaseDropoutBlock)

local defaultDropoutRate = 0.5

function DropoutBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDropoutBlock = BaseDropoutBlock.new()
	
	setmetatable(NewDropoutBlock, DropoutBlock)
	
	NewDropoutBlock:setName("Dropout")
	
	local dropoutRate = parameterDictionary.dropoutRate or defaultDropoutRate
	
	NewDropoutBlock.dropoutRate = dropoutRate

	NewDropoutBlock:setFunction(function(inputTensorArray)
		
		local nonDropoutRate = 1 - NewDropoutBlock.dropoutRate
		
		local scalingFactor = 1 / nonDropoutRate
		
		local functionToApply = function (x)
			
			local isDroppedOut = (math.random() > nonDropoutRate)
			
			return (isDroppedOut and 0) or (x * scalingFactor)
			
		end
		
		return AqwamTensorLibrary:applyFunction(functionToApply, inputTensorArray[1])
		
	end)

	NewDropoutBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		return {initialPartialFirstDerivativeTensor}
		
	end)
	
	return NewDropoutBlock
	
end

function DropoutBlock:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.dropoutRate = parameterDictionary.dropoutRate or self.dropoutRate
	
end

return DropoutBlock
