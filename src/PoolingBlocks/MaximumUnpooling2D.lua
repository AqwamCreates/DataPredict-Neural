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

local MaximumUnpooling2DBlock = {}

MaximumUnpooling2DBlock.__index = MaximumUnpooling2DBlock

setmetatable(MaximumUnpooling2DBlock, BasePoolingBlock)

local defaultKernelDimensionSizeArray = {2, 2}

local defaultStrideDimensionSizeArray = {1, 1}

local defaultUnpoolingMethod = "NearestNeighbour"

local unpoolingMethodFunctionList = {

	["BedOfNails"] = function(tensor, value, a, b, startC, startD, endC, endD)

		tensor[a][b][startC][startD] = value 

	end,

	["NearestNeighbour"] = function(tensor, value, a, b, startC, startD, endC, endD)

		for c = startC, endC, 1 do

			for d = startD, endD, 1 do

				tensor[a][b][c][d] = value 

			end

		end

	end,

}

local unpoolingMethodInverseFunctionList = {

	["BedOfNails"] = function(tensor, otherTensor, a, b, c, d, startC, startD, endC, endD)

		tensor[a][b][c][d] = otherTensor[startC][startD]

	end,

	["NearestNeighbour"] = function(tensor, otherTensor, a, b, c, d, startC, startD, endC, endD)

		for x = startC, endC, 1 do

			for y = startD, endD, 1 do

				tensor[a][b][c][d] = tensor[a][b][c][d] + otherTensor[a][b][x][y]

			end

		end

	end,

}

function MaximumUnpooling2DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMaximumUnpooling2DBlock = BasePoolingBlock.new()

	setmetatable(NewMaximumUnpooling2DBlock, MaximumUnpooling2DBlock)

	NewMaximumUnpooling2DBlock:setName("MaximumUnpooling2D")

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or defaultKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or defaultStrideDimensionSizeArray

	local unpoolingMethod = parameterDictionary.unpoolingMethod or defaultUnpoolingMethod

	if (#kernelDimensionSizeArray ~= 2) then error("The number of dimensions for the kernel dimension size array does not equal to 2.") end

	if (#strideDimensionSizeArray ~= 2) then error("The number of dimensions for the stride dimension size array does not equal to 2.") end

	NewMaximumUnpooling2DBlock.kernelDimensionSizeArray = kernelDimensionSizeArray

	NewMaximumUnpooling2DBlock.strideDimensionSizeArray = strideDimensionSizeArray

	NewMaximumUnpooling2DBlock.unpoolingMethod = unpoolingMethod

	NewMaximumUnpooling2DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial maximum unpooling function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSizeArray = NewMaximumUnpooling2DBlock.kernelDimensionSizeArray 

		local strideDimensionSizeArray = NewMaximumUnpooling2DBlock.strideDimensionSizeArray

		local unpoolingMethodFunction = unpoolingMethodFunctionList[NewMaximumUnpooling2DBlock.unpoolingMethod]

		if (not unpoolingMethodFunction) then error("Invalid unpooling method.") end

		local transformedTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

		for dimension = 1, 2, 1 do

			local inputDimensionSize = inputTensorDimensionSizeArray[dimension + 2]

			local outputDimensionSize = (inputDimensionSize - 1) * strideDimensionSizeArray[dimension] + kernelDimensionSizeArray[dimension]

			transformedTensorDimensionSizeArray[dimension + 2] = outputDimensionSize

		end

		local transformedTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray, 0)

		for a = 1, inputTensorDimensionSizeArray[1], 1 do

			for b = 1, inputTensorDimensionSizeArray[2], 1 do

				for c = 1, inputTensorDimensionSizeArray[3], 1 do

					for d = 1, inputTensorDimensionSizeArray[4], 1 do

						local value = inputTensor[a][b][c][d]

						local startC = (c - 1) * strideDimensionSizeArray[1] + 1
						local startD = (d - 1) * strideDimensionSizeArray[2] + 1

						local endC = startC + kernelDimensionSizeArray[1] - 1
						local endD = startD + kernelDimensionSizeArray[2] - 1

						unpoolingMethodFunction(transformedTensor, value, a, b, c, d, startC, startD, endC, endD)

					end

				end

			end

		end

		return transformedTensor

	end)

	NewMaximumUnpooling2DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSizeArray = NewMaximumUnpooling2DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewMaximumUnpooling2DBlock.strideDimensionSizeArray

		local unpoolingMethodInverseFunction = unpoolingMethodInverseFunctionList[NewMaximumUnpooling2DBlock.unpoolingMethod]

		if (not unpoolingMethodInverseFunction) then error("Invalid unpooling method.") end

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		for a = 1, inputTensorDimensionSizeArray[1], 1 do

			for b = 1, inputTensorDimensionSizeArray[2], 1 do

				for c = 1, inputTensorDimensionSizeArray[3], 1 do

					for d = 1, inputTensorDimensionSizeArray[4], 1 do

						local startC = (c - 1) * strideDimensionSizeArray[1] + 1
						local startD = (d - 1) * strideDimensionSizeArray[2] + 1

						local endC = startC + kernelDimensionSizeArray[1] - 1
						local endD = startD + kernelDimensionSizeArray[2] - 1

						unpoolingMethodInverseFunction(chainRuleFirstDerivativeTensor, initialPartialFirstDerivativeTensor, a, b, c, d, startC, startD, endC, endD)

					end

				end

			end

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewMaximumUnpooling2DBlock

end

return MaximumUnpooling2DBlock
