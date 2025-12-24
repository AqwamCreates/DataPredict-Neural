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

local DropoutNDBlock = {}

DropoutNDBlock.__index = DropoutNDBlock

setmetatable(DropoutNDBlock, BaseDropoutBlock)

local defaultDropoutRate = 0.5

function DropoutNDBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDropoutNDBlock = BaseDropoutBlock.new()
	
	setmetatable(NewDropoutNDBlock, DropoutNDBlock)
	
	NewDropoutNDBlock:setName("DropoutND")
	
	local dropoutRate = parameterDictionary.dropoutRate or defaultDropoutRate
	
	NewDropoutNDBlock.dropoutRate = dropoutRate

	NewDropoutNDBlock:setFunction(function(inputTensorArray)
		
		local inputTensor = inputTensorArray[1]
		
		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray
		
		local numberOfData = inputTensorDimensionSizeArray[1]
		
		local dimensionSizeAtSecondDimension = inputTensorDimensionSizeArray[2]
		
		local dropoutTensorDimensionSizeArray = {}
		
		local nonDropoutRate = 1 - NewDropoutNDBlock.dropoutRate
		
		local scalingFactor = 1 / nonDropoutRate
		
		local transformedTensor = AqwamTensorLibrary:copy(inputTensor)
		
		for i = 3, numberOfDimensions, 1 do table.insert(dropoutTensorDimensionSizeArray, inputTensorDimensionSizeArray[i]) end
		
		for i = 1, numberOfData, 1 do
			
			for j = 1, dimensionSizeAtSecondDimension, 1 do
				
				if (math.random() > nonDropoutRate) then transformedTensor[i][j] = AqwamTensorLibrary:createTensor(dropoutTensorDimensionSizeArray, 0) end
				
			end
			
		end
		
		transformedTensor = AqwamTensorLibrary:multiply(transformedTensor, scalingFactor)
		
		return transformedTensor
		
	end)

	NewDropoutNDBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		return {initialPartialFirstDerivativeTensor}
		
	end)
	
	return NewDropoutNDBlock
	
end

function DropoutNDBlock:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.dropoutRate = parameterDictionary.dropoutRate or self.dropoutRate
	
end

return DropoutNDBlock
