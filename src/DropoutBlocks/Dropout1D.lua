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

Dropout1DBlock = {}

Dropout1DBlock.__index = Dropout1DBlock

setmetatable(Dropout1DBlock, BaseDropoutBlock)

local defaultDropoutRate = 0.5

function Dropout1DBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDropout1DBlock = BaseDropoutBlock.new()
	
	setmetatable(NewDropout1DBlock, Dropout1DBlock)
	
	NewDropout1DBlock:setName("Dropout1D")
	
	local dropoutRate = parameterDictionary.dropoutRate or defaultDropoutRate
	
	NewDropout1DBlock.dropoutRate = dropoutRate

	NewDropout1DBlock:setFunction(function(inputTensorArray)
		
		local inputTensor = inputTensorArray[1]
		
		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray
		
		if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial dropout function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end
		
		local numberOfData = inputTensorDimensionSizeArray[1]
		
		local dimensionSizeAtSecondDimension = inputTensorDimensionSizeArray[2]
		
		local dimensionSizeAtThirdDimension = inputTensorDimensionSizeArray[3]
		
		local nonDropoutRate = 1 - NewDropout1DBlock.dropoutRate
		
		local scalingFactor = 1 / nonDropoutRate
		
		local transformedTensor = AqwamTensorLibrary:copy(inputTensor)
		
		for i = 1, numberOfData, 1 do
			
			for j = 1, dimensionSizeAtSecondDimension, 1 do
				
				if (math.random() > nonDropoutRate) then transformedTensor[i][j] = table.create(dimensionSizeAtThirdDimension, 0) end
				
			end
			
		end
		
		transformedTensor = AqwamTensorLibrary:multiply(transformedTensor, scalingFactor)
		
		return transformedTensor
		
	end)

	NewDropout1DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		return {initialPartialFirstDerivativeTensor}
		
	end)
	
	return NewDropout1DBlock
	
end

function Dropout1DBlock:setParameters(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	self.dropoutRate = parameterDictionary.dropoutRate or self.dropoutRate
	
end

return Dropout1DBlock