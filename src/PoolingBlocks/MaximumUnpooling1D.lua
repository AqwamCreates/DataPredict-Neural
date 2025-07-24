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

local BasePoolingBlock = require(script.Parent.BasePoolingBlock)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

MaximumUnpooling1DBlock = {}

MaximumUnpooling1DBlock.__index = MaximumUnpooling1DBlock

setmetatable(MaximumUnpooling1DBlock, BasePoolingBlock)

local defaultKernelDimensionSize = 2

local defaultStrideDimensionSize = 1

local defaultUnpoolingMethod = "NearestNeighbour"

local unpoolingMethodFunctionList = {

	["BedOfNails"] = function(tensor, value, a, b, startC, endC)

		tensor[a][b][startC] = value 

	end,

	["NearestNeighbour"] = function(tensor, value, a, b, startC, endC)

		for c = startC, endC, 1 do

			tensor[a][b][c] = value 

		end

	end,

}

local unpoolingMethodInverseFunctionList = {

	["BedOfNails"] = function(tensor, otherTensor, a, b, c, startC, endC)

		tensor[a][b][c] = otherTensor[startC]

	end,

	["NearestNeighbour"] = function(tensor, otherTensor, a, b, c, startC, endC)

		for x = startC, endC, 1 do

			tensor[a][b][c] = tensor[a][b][c] + otherTensor[a][b][x]

		end

	end,

}

function MaximumUnpooling1DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMaximumUnpooling1DBlock = BasePoolingBlock.new()

	setmetatable(NewMaximumUnpooling1DBlock, MaximumUnpooling1DBlock)

	NewMaximumUnpooling1DBlock:setName("MaximumUnpooling1D")

	local kernelDimensionSize = parameterDictionary.kernelDimensionSize or defaultKernelDimensionSize

	local strideDimensionSize = parameterDictionary.strideDimensionSize or defaultStrideDimensionSize

	local unpoolingMethod = parameterDictionary.unpoolingMethod or defaultUnpoolingMethod

	if (type(kernelDimensionSize) ~= "number") then error("The kernel dimension size must be a number.") end

	if (type(strideDimensionSize) ~= "number") then error("The stride dimension size must be a number.") end

	NewMaximumUnpooling1DBlock.kernelDimensionSize = kernelDimensionSize

	NewMaximumUnpooling1DBlock.strideDimensionSize = strideDimensionSize

	NewMaximumUnpooling1DBlock.unpoolingMethod = unpoolingMethod

	NewMaximumUnpooling1DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial maximum unpooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSize = NewMaximumUnpooling1DBlock.kernelDimensionSize 

		local strideDimensionSize = NewMaximumUnpooling1DBlock.strideDimensionSize

		local unpoolingMethodFunction = unpoolingMethodFunctionList[NewMaximumUnpooling1DBlock.unpoolingMethod]

		if (not unpoolingMethodFunction) then error("Invalid unpooling method.") end

		local transformedTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

		local inputDimensionSize = inputTensorDimensionSizeArray[3]

		local outputDimensionSize = (inputDimensionSize - 1) * strideDimensionSize + kernelDimensionSize

		transformedTensorDimensionSizeArray[3] = outputDimensionSize

		local transformedTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray, 0)

		for a = 1, inputTensorDimensionSizeArray[1], 1 do

			for b = 1, inputTensorDimensionSizeArray[2], 1 do

				for c = 1, inputTensorDimensionSizeArray[3], 1 do

					local value = inputTensor[a][b][c]

					local startC = (c - 1) * strideDimensionSize + 1

					local endC = startC + kernelDimensionSize - 1

					unpoolingMethodFunction(transformedTensor, value, a, b, c, startC, endC)

				end

			end

		end

		return transformedTensor

	end)

	NewMaximumUnpooling1DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSize = NewMaximumUnpooling1DBlock.kernelDimensionSize

		local strideDimensionSize = NewMaximumUnpooling1DBlock.strideDimensionSize

		local unpoolingMethodInverseFunction = unpoolingMethodInverseFunctionList[NewMaximumUnpooling1DBlock.unpoolingMethod]

		if (not unpoolingMethodInverseFunction) then error("Invalid unpooling method.") end

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		for a = 1, inputTensorDimensionSizeArray[1], 1 do

			for b = 1, inputTensorDimensionSizeArray[2], 1 do

				for c = 1, inputTensorDimensionSizeArray[3], 1 do

					local startC = (c - 1) * strideDimensionSize + 1

					local endC = startC + kernelDimensionSize - 1

					unpoolingMethodInverseFunction(chainRuleFirstDerivativeTensor, initialPartialFirstDerivativeTensor, a, b, c, startC, endC)

				end

			end

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewMaximumUnpooling1DBlock

end

return MaximumUnpooling1DBlock