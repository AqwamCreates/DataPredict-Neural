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

MaximumUnpooling3DBlock = {}

MaximumUnpooling3DBlock.__index = MaximumUnpooling3DBlock

setmetatable(MaximumUnpooling3DBlock, BasePoolingBlock)

local defaultKernelDimensionSizeArray = {2, 2, 2}

local defaultStrideDimensionSizeArray = {1, 1, 1}

local defaultUnpoolingMethod = "NearestNeighbour"

local unpoolingMethodFunctionList = {

	["BedOfNails"] = function(tensor, value, a, b, startC, startD, startE, endC, endD, endE)

		tensor[a][b][startC][startD][startE] = value 

	end,

	["NearestNeighbour"] = function(tensor, value, a, b, startC, startD, startE, endC, endD, endE)

		for c = startC, endC, 1 do

			for d = startD, endD, 1 do

				for e = startE, endE, 1 do

					tensor[a][b][c][d][e] = value 

				end

			end

		end

	end,

}

local unpoolingMethodInverseFunctionList = {

	["BedOfNails"] = function(tensor, otherTensor, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

		tensor[a][b][c][d][e] = otherTensor[startC][startD][startE]

	end,

	["NearestNeighbour"] = function(tensor, otherTensor, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

		for x = startC, endC, 1 do

			for y = startD, endD, 1 do

				for z = startE, endE, 1 do

					tensor[a][b][c][d][e] = tensor[a][b][c][d][e] + otherTensor[a][b][x][y][z]

				end

			end

		end

	end,

}

function MaximumUnpooling3DBlock.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMaximumUnpooling3DBlock = BasePoolingBlock.new()

	setmetatable(NewMaximumUnpooling3DBlock, MaximumUnpooling3DBlock)

	NewMaximumUnpooling3DBlock:setName("MaximumUnpooling3D")

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or defaultKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or defaultStrideDimensionSizeArray

	local unpoolingMethod = parameterDictionary.unpoolingMethod or defaultUnpoolingMethod

	if (#kernelDimensionSizeArray ~= 3) then error("The number of dimensions for the kernel dimension size array does not equal to 3.") end

	if (#strideDimensionSizeArray ~= 3) then error("The number of dimensions for the stride dimension size array does not equal to 3.") end

	NewMaximumUnpooling3DBlock.kernelDimensionSizeArray = kernelDimensionSizeArray

	NewMaximumUnpooling3DBlock.strideDimensionSizeArray = strideDimensionSizeArray

	NewMaximumUnpooling3DBlock.unpoolingMethod = unpoolingMethod

	NewMaximumUnpooling3DBlock:setFunction(function(inputTensorArray)

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)
		
		local numberOfDimensions = #inputTensorDimensionSizeArray

		if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial maximum unpooling function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

		local kernelDimensionSizeArray = NewMaximumUnpooling3DBlock.kernelDimensionSizeArray 

		local strideDimensionSizeArray = NewMaximumUnpooling3DBlock.strideDimensionSizeArray

		local unpoolingMethodFunction = unpoolingMethodFunctionList[NewMaximumUnpooling3DBlock.unpoolingMethod]

		if (not unpoolingMethodFunction) then error("Invalid unpooling method.") end

		local transformedTensorDimensionSizeArray = table.clone(inputTensorDimensionSizeArray)

		for dimension = 1, 3, 1 do

			local inputDimensionSize = inputTensorDimensionSizeArray[dimension + 2]

			local outputDimensionSize = (inputDimensionSize - 1) * strideDimensionSizeArray[dimension] + kernelDimensionSizeArray[dimension]

			transformedTensorDimensionSizeArray[dimension + 2] = outputDimensionSize

		end

		local transformedTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray, 0)

		for a = 1, inputTensorDimensionSizeArray[1], 1 do

			for b = 1, inputTensorDimensionSizeArray[2], 1 do

				for c = 1, inputTensorDimensionSizeArray[3], 1 do

					for d = 1, inputTensorDimensionSizeArray[4], 1 do

						for e = 1, inputTensorDimensionSizeArray[5], 1 do

							local value = inputTensor[a][b][c][d][e]

							local startC = (c - 1) * strideDimensionSizeArray[1] + 1
							local startD = (d - 1) * strideDimensionSizeArray[2] + 1
							local startE = (e - 1) * strideDimensionSizeArray[3] + 1

							local endC = startC + kernelDimensionSizeArray[1] - 1
							local endD = startD + kernelDimensionSizeArray[2] - 1
							local endE = startE + kernelDimensionSizeArray[3] - 1

							unpoolingMethodFunction(transformedTensor, value, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

						end

					end

				end

			end

		end

		return transformedTensor

	end)

	NewMaximumUnpooling3DBlock:setChainRuleFirstDerivativeFunction(function(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

		local kernelDimensionSizeArray = NewMaximumUnpooling3DBlock.kernelDimensionSizeArray

		local strideDimensionSizeArray = NewMaximumUnpooling3DBlock.strideDimensionSizeArray

		local unpoolingMethodInverseFunction = unpoolingMethodInverseFunctionList[NewMaximumUnpooling3DBlock.unpoolingMethod]

		if (not unpoolingMethodInverseFunction) then error("Invalid unpooling method.") end

		local inputTensor = inputTensorArray[1]

		local inputTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(inputTensorDimensionSizeArray)

		for a = 1, inputTensorDimensionSizeArray[1], 1 do

			for b = 1, inputTensorDimensionSizeArray[2], 1 do

				for c = 1, inputTensorDimensionSizeArray[3], 1 do

					for d = 1, inputTensorDimensionSizeArray[4], 1 do

						for e = 1, inputTensorDimensionSizeArray[5], 1 do

							local startC = (c - 1) * strideDimensionSizeArray[1] + 1
							local startD = (d - 1) * strideDimensionSizeArray[2] + 1
							local startE = (e - 1) * strideDimensionSizeArray[3] + 1

							local endC = startC + kernelDimensionSizeArray[1] - 1
							local endD = startD + kernelDimensionSizeArray[2] - 1
							local endE = startE + kernelDimensionSizeArray[3] - 1

							unpoolingMethodInverseFunction(chainRuleFirstDerivativeTensor, initialPartialFirstDerivativeTensor, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

						end

					end

				end

			end

		end

		return {chainRuleFirstDerivativeTensor}

	end)

	return NewMaximumUnpooling3DBlock

end

return MaximumUnpooling3DBlock