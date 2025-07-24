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

PairwiseDistanceBlock = {}

PairwiseDistanceBlock.__index = PairwiseDistanceBlock

setmetatable(PairwiseDistanceBlock, BaseOperatorBlock)

local defaultP = 2

function PairwiseDistanceBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewPairwiseDistanceBlock = BaseOperatorBlock.new()

	setmetatable(NewPairwiseDistanceBlock, PairwiseDistanceBlock)

	NewPairwiseDistanceBlock:setName("PairwiseDistance")

	NewPairwiseDistanceBlock:setFirstDerivativeFunctionRequiresTransformedTensor(false)
	
	NewPairwiseDistanceBlock.p = parameterDictionary.p or defaultP
	
	NewPairwiseDistanceBlock.distanceTensor = nil

	NewPairwiseDistanceBlock:setFunction(function(inputTensorArray)
		
		local p = NewPairwiseDistanceBlock.p
		
		local subtractedTensor = AqwamTensorLibrary:subtract(table.unpack(inputTensorArray))
		
		local powerSubtractedTensor = AqwamTensorLibrary:power(subtractedTensor, p)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(powerSubtractedTensor)
		
		local numberOfDimensions = #dimensionSizeArray
		
		local distanceTensor = powerSubtractedTensor
		
		for dimension = numberOfDimensions, 2, -1 do
			
			distanceTensor = AqwamTensorLibrary:sum(distanceTensor, dimension)
			
		end
		
		NewPairwiseDistanceBlock.distanceTensor = distanceTensor

		return distanceTensor

	end)

	NewPairwiseDistanceBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
		
		local inputTensor1 = inputTensorArray[1]

		local inputTensor2 = inputTensorArray[2]
		
		local p = NewPairwiseDistanceBlock.p
		
		local distanceTensor = NewPairwiseDistanceBlock.distanceTensor
		
		local powerDistanceTensor = AqwamTensorLibrary:power(distanceTensor, ((1 / p) - 1))
		
		local subtractedTensor = AqwamTensorLibrary:subtract(inputTensor1, inputTensor2)
		
		local absoluteSubtractedTensor = AqwamTensorLibrary:applyFunction(math.abs, subtractedTensor)
		
		local powerAbsoulteSubtractedTensor = AqwamTensorLibrary:power(absoluteSubtractedTensor, (p - 1))
		
		local signSubtractedTensor = AqwamTensorLibrary:applyFunction(math.sign, subtractedTensor)
		
		local chainRuleFirstDerivativeTensor1 = AqwamTensorLibrary:multiply(initialPartialFirstDerivativeTensor, powerDistanceTensor, powerAbsoulteSubtractedTensor, signSubtractedTensor)
		
		local chainRuleFirstDerivativeTensor2 = AqwamTensorLibrary:unaryMinus(chainRuleFirstDerivativeTensor1)
		
		local dimensionSizeArray1 = AqwamTensorLibrary:getDimensionSizeArray(inputTensor1)
		
		local dimensionSizeArray2 = AqwamTensorLibrary:getDimensionSizeArray(inputTensor2)
		
		chainRuleFirstDerivativeTensor1 = NewPairwiseDistanceBlock:collapseTensor(chainRuleFirstDerivativeTensor1, dimensionSizeArray1)
		
		chainRuleFirstDerivativeTensor2 = NewPairwiseDistanceBlock:collapseTensor(chainRuleFirstDerivativeTensor2, dimensionSizeArray2)

		return {chainRuleFirstDerivativeTensor1, chainRuleFirstDerivativeTensor2}

	end)

	return NewPairwiseDistanceBlock

end

return PairwiseDistanceBlock