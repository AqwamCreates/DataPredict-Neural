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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseWeightBlock = require(script.Parent.Parent.WeightBlocks.BaseWeightBlock)

BaseConvolutionBlock = {}

BaseConvolutionBlock.__index = BaseConvolutionBlock

setmetatable(BaseConvolutionBlock, BaseWeightBlock)

local defaultLearningRate = 0.01

local defaultWeightInitializationMode = "RandomUniformNegativeAndPositive"

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

end

function BaseConvolutionBlock.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewBaseConvolutionBlock = BaseWeightBlock.new(parameterDictionary)

	setmetatable(NewBaseConvolutionBlock, BaseConvolutionBlock)

	NewBaseConvolutionBlock:setName("BaseConvolutionBlock")

	NewBaseConvolutionBlock:setClassName("ConvolutionBlock")

	NewBaseConvolutionBlock:setSaveInputTensorArray(true)

	NewBaseConvolutionBlock:setSaveTransformedTensor(true)

	NewBaseConvolutionBlock:setSaveTotalFirstDerivativeTensorArray(true)

	return NewBaseConvolutionBlock

end

local function waitUntilAllCoroutinesFinished(coroutineArray)

	while true do

		local allFinished = true

		for _, coroutineInstance in ipairs(coroutineArray) do

			if coroutine.status(coroutineInstance) ~= "dead" then

				allFinished = false

				break

			end

		end

		if allFinished then break end

		task.wait()

	end

end

function BaseConvolutionBlock:createCoroutineToArray(coroutineArray, functionToRun)

	local oneCoroutine = coroutine.create(functionToRun)

	table.insert(coroutineArray, oneCoroutine)

end

function BaseConvolutionBlock:runCoroutinesUntilFinished(coroutineArray)

	for _, oneCoroutine in coroutineArray do coroutine.resume(oneCoroutine) end

	waitUntilAllCoroutinesFinished(coroutineArray)

end

return BaseConvolutionBlock