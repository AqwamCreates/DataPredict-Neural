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

local BaseAttentionBlock = require(script.Parent.BaseAttentionBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

ScaledDotProductAttentionBlock = {}

ScaledDotProductAttentionBlock.__index = ScaledDotProductAttentionBlock

setmetatable(ScaledDotProductAttentionBlock, BaseAttentionBlock)

function ScaledDotProductAttentionBlock.new()

	local NewScaledDotProductAttentionBlock = BaseAttentionBlock.new()

	setmetatable(NewScaledDotProductAttentionBlock, ScaledDotProductAttentionBlock)

	NewScaledDotProductAttentionBlock:setName("ScaledDotProductAttention")

	local softmaxTensor

	NewScaledDotProductAttentionBlock:setFunction(function(inputTensorArray)

		local queryTensor = inputTensorArray[1]

		local keyTensor = inputTensorArray[2]

		local valueTensor = inputTensorArray[3]

		local keyDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(keyTensor)

		local finalDimensionSize = keyDimensionSizeArray[#keyDimensionSizeArray]

		local keyTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(keyTensor)

		local transposedKeyTensor = AqwamTensorLibrary:transpose(keyTensor, {keyTensorNumberOfDimensions - 1, keyTensorNumberOfDimensions})

		local dotProductQueryAndKeyTensor = AqwamTensorLibrary:dotProduct(queryTensor, transposedKeyTensor)

		local squareRootFinalDimensionSize = math.sqrt(finalDimensionSize)

		local scaledDotProductQueryAndKeyTensor = AqwamTensorLibrary:divide(dotProductQueryAndKeyTensor, squareRootFinalDimensionSize)

		local exponentInputTensor = AqwamTensorLibrary:applyFunction(math.exp, scaledDotProductQueryAndKeyTensor)

		local summedExponentInputTensor = AqwamTensorLibrary:sum(exponentInputTensor, 2)

		softmaxTensor = AqwamTensorLibrary:divide(exponentInputTensor, summedExponentInputTensor)

		local transformedTensor = AqwamTensorLibrary:dotProduct(softmaxTensor, valueTensor)

		return transformedTensor

	end)

	NewScaledDotProductAttentionBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local queryTensor = inputTensorArray[1]

		local keyTensor = inputTensorArray[2]

		local valueTensor = inputTensorArray[3]

		----------------------------------------------------------------------

		local softmaxTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(softmaxTensor)

		local valueTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(valueTensor)

		local softMaxInitialPartialFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(initialPartialFirstDerivativeTensor, AqwamTensorLibrary:transpose(valueTensor, {valueTensorNumberOfDimensions - 1, valueTensorNumberOfDimensions}))
		
		local valuePartialFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(softmaxTensor, {softmaxTensorNumberOfDimensions - 1, softmaxTensorNumberOfDimensions}), initialPartialFirstDerivativeTensor)

		---------------------------------------------------------

		local softmaxDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(softMaxInitialPartialFirstDerivativeTensor)

		local softmaxPartialFirstDerivativeTensor = AqwamTensorLibrary:createTensor(softmaxDimensionSizeArray, 0)

		local numberOfRows = softmaxDimensionSizeArray[1]

		local numberOfColumns = softmaxDimensionSizeArray[2]

		for i = 1, numberOfRows, 1 do

			for j = 1, numberOfColumns do

				for k = 1, numberOfColumns do

					if (j == k) then

						softmaxPartialFirstDerivativeTensor[i][j] = softmaxPartialFirstDerivativeTensor[i][j] + (softmaxTensor[i][j] * (1 - softmaxTensor[i][k]))

					else

						softmaxPartialFirstDerivativeTensor[i][j] = softmaxPartialFirstDerivativeTensor[i][j] + (-softmaxTensor[i][j] * softmaxTensor[i][k])

					end

				end

			end

		end

		local softmaxFirstDerivativeTensor = AqwamTensorLibrary:multiply(softMaxInitialPartialFirstDerivativeTensor, softmaxPartialFirstDerivativeTensor)

		----------------------------------------------------------------------

		local keyDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(keyTensor)

		local finalDimensionSize = keyDimensionSizeArray[#keyDimensionSizeArray]

		local squareRootFinalDimensionSize = math.sqrt(finalDimensionSize)

		local scaledDotProductQueryAndKeyFirstDerivativeTensor = AqwamTensorLibrary:divide(softmaxFirstDerivativeTensor, squareRootFinalDimensionSize)

		local transposedKeyTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(softMaxInitialPartialFirstDerivativeTensor)

		local transposedKeyTensor = AqwamTensorLibrary:transpose(keyTensor, {transposedKeyTensorNumberOfDimensions - 1, transposedKeyTensorNumberOfDimensions})

		local queryTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(queryTensor)

		local transposedKeyFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(queryTensor, {queryTensorNumberOfDimensions - 1, queryTensorNumberOfDimensions}), scaledDotProductQueryAndKeyFirstDerivativeTensor)

		local queryPartialFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(scaledDotProductQueryAndKeyFirstDerivativeTensor, keyTensor)

		local transposedKeyFirstDerivativeTensorNumberOfDimensions = AqwamTensorLibrary:getNumberOfDimensions(transposedKeyFirstDerivativeTensor)

		local keyPartialFirstDerivativeTensor = AqwamTensorLibrary:transpose(transposedKeyFirstDerivativeTensor, {transposedKeyFirstDerivativeTensorNumberOfDimensions - 1, transposedKeyFirstDerivativeTensorNumberOfDimensions})

		return {queryPartialFirstDerivativeTensor, keyPartialFirstDerivativeTensor, valuePartialFirstDerivativeTensor}

	end)

	return NewScaledDotProductAttentionBlock

end

return ScaledDotProductAttentionBlock