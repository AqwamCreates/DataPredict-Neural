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

local Dropout2DBlock = {}

Dropout2DBlock.__index = Dropout2DBlock

setmetatable(Dropout2DBlock, BaseDropoutBlock)

local defaultDropoutRate = 0.5

function Dropout2DBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDropout2DBlock = BaseDropoutBlock.new()
	
	setmetatable(NewDropout2DBlock, Dropout2DBlock)
	
	NewDropout2DBlock:setName("Dropout2D")
	
	local dropoutRate = parameterDictionary.dropoutRate or defaultDropoutRate
	
	NewDropout2DBlock.dropoutRate = dropoutRate

	NewDropout2DBlock:setFunction(function(inputTensorArray)
		
		local inputTensor = inputTensorArray[1]
		
		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray
		
		if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial dropout function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end
		
		local numberOfData = inputTensorDimensionSizeArray[1]
		
		local dimensionSizeAtSecondDimension = inputTensorDimensionSizeArray[2]
		
		local dropoutTensorDimensionSizeArray = {inputTensorDimensionSizeArray[3], inputTensorDimensionSizeArray[4]}
		
		local nonDropoutRate = 1 - NewDropout2DBlock.dropoutRate
		
		local scalingFactor = 1 / nonDropoutRate
		
		local transformedTensor = AqwamTensorLibrary:copy(inputTensor)
		
		for i = 1, numberOfData, 1 do
			
			for j = 1, dimensionSizeAtSecondDimension, 1 do
				
				if (math.random() > nonDropoutRate) then transformedTensor[i][j] = AqwamTensorLibrary:createTensor(dropoutTensorDimensionSizeArray, 0) end
				
			end
			
		end
		
		transformedTensor = AqwamTensorLibrary:multiply(transformedTensor, scalingFactor)
		
		return transformedTensor
		
	end)

	NewDropout2DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		return {initialPartialFirstDerivativeTensor}
		
	end)
	
	return NewDropout2DBlock
	
end

function Dropout2DBlock:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.dropoutRate = parameterDictionary.dropoutRate or self.dropoutRate
	
end

return Dropout2DBlock
