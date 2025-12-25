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

local BaseOperatorBlock = require(script.Parent.BaseOperatorBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ConcatenateBlock = {}

ConcatenateBlock.__index = ConcatenateBlock

setmetatable(ConcatenateBlock, BaseOperatorBlock)

function ConcatenateBlock.new(parameterDictionary)

	local NewConcatenateBlock = BaseOperatorBlock.new()

	setmetatable(NewConcatenateBlock, ConcatenateBlock)

	NewConcatenateBlock:setName("Concatenate")
	
	NewConcatenateBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	parameterDictionary = parameterDictionary or {}
	
	local dimension = parameterDictionary.dimension
	
	if (not dimension) then error("No dimension for concatenating along an axis!") end
	
	NewConcatenateBlock.dimension = dimension
	
	NewConcatenateBlock:setFunction(function(inputTensorArray)
		
		local transformedTensor
		
		local dimension = NewConcatenateBlock.dimension
		
		for i, inputTensor in ipairs(inputTensorArray) do
			
			if (i > 1) then
				
				transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, inputTensor, dimension)
				
			else
				
				transformedTensor = inputTensor
				
			end
			
			
		end
		
		return transformedTensor
	
	end)

	NewConcatenateBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local chainRuleFirstDerivativeTensorArray = {}
		
		local dimension = NewConcatenateBlock.dimension
		
		local initialPartialFirstDerivativeTensorDimensionArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)
		
		local originDimensionIndexArray = table.create(#initialPartialFirstDerivativeTensorDimensionArray, 1)
		
		local targetDimensionIndexArray = table.clone(initialPartialFirstDerivativeTensorDimensionArray)
		
		targetDimensionIndexArray[dimension] = 0
		
		for _, inputTensor in ipairs(inputTensorArray) do
			
			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
			
			targetDimensionIndexArray[dimension] = originDimensionIndexArray[dimension] + dimensionSizeArray[dimension] - 1
			
			local firstDerivativeTensor = AqwamTensorLibrary:extract(initialPartialFirstDerivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)
			
			originDimensionIndexArray[dimension] = originDimensionIndexArray[dimension] + dimensionSizeArray[dimension]
			
			table.insert(chainRuleFirstDerivativeTensorArray, firstDerivativeTensor)
			
		end

		return chainRuleFirstDerivativeTensorArray
		
	end)

	return NewConcatenateBlock

end

function ConcatenateBlock:setParameters(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	self.dimension = parameterDictionary.dimension or self.dimension

end

return ConcatenateBlock
