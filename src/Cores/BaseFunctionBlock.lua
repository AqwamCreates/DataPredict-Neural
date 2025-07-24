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

local BaseInstance = require(script.Parent.BaseInstance)

BaseFunctionBlock = {}

BaseFunctionBlock.__index = BaseFunctionBlock

setmetatable(BaseFunctionBlock, BaseInstance)

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

local function waitForValue(waitDuration, functionToRun)

	waitDuration = waitDuration or math.huge

	local currentDuration = 0

	local value = functionToRun()

	while (not value) and (currentDuration < waitDuration) do

		currentDuration = currentDuration + wait()

		value = functionToRun()

	end

end

local function getValueOrDefaultValue(value, defaultValue)

	if (type(value) == "nil") then return defaultValue end

	return value

end

function BaseFunctionBlock.new()

	local NewBaseFunctionBlock = BaseInstance.new()

	setmetatable(NewBaseFunctionBlock, BaseFunctionBlock)

	NewBaseFunctionBlock:setName("BaseFunctionBlock")

	NewBaseFunctionBlock:setClassName("FunctionBlock")

	NewBaseFunctionBlock.Function = nil

	NewBaseFunctionBlock.ChainRuleFirstDerivativeFunction = nil

	NewBaseFunctionBlock.FirstDerivativeFunction = nil

	NewBaseFunctionBlock.saveInputTensorArray = false

	NewBaseFunctionBlock.saveTransformedTensor = false

	NewBaseFunctionBlock.saveTotalChainRuleFirstDerivativeTensorArray = false

	NewBaseFunctionBlock.saveTotalFirstDerivativeTensorArray = false

	NewBaseFunctionBlock.requiresInputTensors = true
	
	NewBaseFunctionBlock.chainRuleFirstDerivativeFunctionRequiresTransformedTensor = false

	NewBaseFunctionBlock.firstDerivativeFunctionRequiresTransformedTensor = false

	NewBaseFunctionBlock.waitForAllInitialPartialFirstDerivativeTensors = true

	NewBaseFunctionBlock.NextFunctionBlockArray = {}

	NewBaseFunctionBlock.PreviousFunctionBlockArray = {}

	NewBaseFunctionBlock.inputTensorArrayToUse = {}

	NewBaseFunctionBlock.inputTensorArray = {}

	NewBaseFunctionBlock.initialPartialFirstDerivativeTensorArray = {}

	NewBaseFunctionBlock.totalNumberOfDifferentiateFunctionCall = 0

	NewBaseFunctionBlock.totalInitialPartialFirstDerivativeTensor = nil
	
	NewBaseFunctionBlock.transformedTensorDimensionSizeArray = nil

	return NewBaseFunctionBlock

end

function BaseFunctionBlock:setFunction(Function)

	self.Function = Function

end

function BaseFunctionBlock:getFunction()

	return self.Function

end

function BaseFunctionBlock:setChainRuleFirstDerivativeFunction(ChainRuleFirstDerivativeFunction)

	self.ChainRuleFirstDerivativeFunction = ChainRuleFirstDerivativeFunction

end

function BaseFunctionBlock:getChainRuleFirstDerivativeFunction()

	return self.ChainRuleFirstDerivativeFunction

end

function BaseFunctionBlock:setFirstDerivativeFunction(FirstDerivativeFunction)

	self.FirstDerivativeFunction = FirstDerivativeFunction

end

function BaseFunctionBlock:getFirstDerivativeFunction()

	return self.FirstDerivativeFunction

end

function BaseFunctionBlock:transformAndSendToNextBlocksIfAllInputTensorsAreReceived()

	local inputTensorArrayToUse = self.inputTensorArrayToUse

	local numberOfPreviousFunctionBlocks = #self.PreviousFunctionBlockArray

	local requiresInputTensors = self.requiresInputTensors

	if (#inputTensorArrayToUse ~= numberOfPreviousFunctionBlocks) and (numberOfPreviousFunctionBlocks ~= 0) and (requiresInputTensors) then return nil end

	if (self.saveInputTensorArray) and (requiresInputTensors) then self.inputTensorArray = deepCopyTable(inputTensorArrayToUse) end

	local transformedTensor = self.Function(inputTensorArrayToUse)
	
	self.transformedTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(transformedTensor)

	table.clear(inputTensorArrayToUse)

	for _, NextFunctionBlock in ipairs(self.NextFunctionBlockArray) do

		task.spawn((function() NextFunctionBlock:internalTransform(self, transformedTensor) end))

	end

	if (self.saveTransformedTensor) then self.transformedTensor = deepCopyTable(transformedTensor) end

	return transformedTensor

end

function BaseFunctionBlock:collapseTensor(tensor, targetDimensionSizeArray)
	
	local numberOfDimensionsOfTensor = #targetDimensionSizeArray

	local numberOfDimensionsOfDerivativeTensor = #AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensionsToSum = numberOfDimensionsOfDerivativeTensor - numberOfDimensionsOfTensor

	for i = 1, numberOfDimensionsToSum, 1 do tensor = AqwamTensorLibrary:sum(tensor, 1)[1] end

	for i, size in ipairs(targetDimensionSizeArray) do

		if (size == 1) then tensor = AqwamTensorLibrary:sum(tensor, i) end

	end

	return tensor
	
end

function BaseFunctionBlock:transform(inputTensor)

	table.insert(self.inputTensorArrayToUse, inputTensor)

	return self:transformAndSendToNextBlocksIfAllInputTensorsAreReceived()

end

function BaseFunctionBlock:internalTransform(SourceBlock, inputTensor)

	local index = self:findPreviousFunctionBlock(SourceBlock)

	self.inputTensorArrayToUse[index] = inputTensor

	return self:transformAndSendToNextBlocksIfAllInputTensorsAreReceived()

end

function BaseFunctionBlock:accumulateChainRuleFirstDerivativeTensorArray(chainRuleFirstDerivativeTensorArray)
	
	if (self.saveTotalChainRuleFirstDerivativeTensorArray) then

		local totalChainRuleFirstDerivativeTensorArray = self.totalChainRuleFirstDerivativeTensorArray

		if (totalChainRuleFirstDerivativeTensorArray) then

			for i, chainRuleFirstDerivativeTensor in ipairs(chainRuleFirstDerivativeTensorArray) do self.totalChainRuleFirstDerivativeTensorArray[i] = AqwamTensorLibrary:add(totalChainRuleFirstDerivativeTensorArray[i], chainRuleFirstDerivativeTensor) end

		else

			self.totalChainRuleFirstDerivativeTensorArray = deepCopyTable(chainRuleFirstDerivativeTensorArray) 

		end

	end
	
end

function BaseFunctionBlock:calculateChainRuleFirstDerivativeTensorArrayAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local chainRuleFirstDerivativeTensorArray = self.ChainRuleFirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) -- Can be considered as partial first derivative.
	
	self:accumulateChainRuleFirstDerivativeTensorArray(chainRuleFirstDerivativeTensorArray)

	for i, PreviousFunctionBlock in ipairs(self.PreviousFunctionBlockArray) do

		task.spawn(function() PreviousFunctionBlock:differentiate(chainRuleFirstDerivativeTensorArray[i]) end)

	end

end

function BaseFunctionBlock:accumulateFirstDerivativeTensorArray(firstDerivativeTensorArray)
	
	if (self.saveTotalFirstDerivativeTensorArray) then 

		local totalFirstDerivativeTensorArray = self.totalFirstDerivativeTensorArray

		if (totalFirstDerivativeTensorArray) then

			for i, firstDerivativeTensor in ipairs(firstDerivativeTensorArray) do self.totalFirstDerivativeTensorArray[i] = AqwamTensorLibrary:add(totalFirstDerivativeTensorArray[i], firstDerivativeTensor) end

		else

			self.totalFirstDerivativeTensorArray = deepCopyTable(firstDerivativeTensorArray) 

		end

	end
	
end

function BaseFunctionBlock:calculateFirstDerivativeTensorArray(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	local firstDerivativeTensorArray = self.FirstDerivativeFunction(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)

	self:accumulateFirstDerivativeTensorArray(firstDerivativeTensorArray)

end

function BaseFunctionBlock:calculateAllFirstDerivativeTensorsAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) -- The code was designed so that when one calculation step requires the transformed tensor, the other calculation step does not need to wait for the transformed tensor if it doesn't need it.
	
	if (self.firstDerivativeFunctionsRequiresTransformedInputTensor) and (self.chainRuleFirstDerivativeFunctionsRequiresTransformedInputTensor) then

		if (not transformedTensor) then transformedTensor = self.Function(inputTensorArray) end
		
		task.spawn(function() self:calculateChainRuleFirstDerivativeTensorArrayAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) end)

		if (self.FirstDerivativeFunction) then task.spawn(function() self:calculateFirstDerivativeTensorArray(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) end) end
		
	elseif (self.firstDerivativeFunctionsRequiresTransformedInputTensor) and (not self.chainRuleFirstDerivativeFunctionsRequiresTransformedInputTensor) then
		
		task.spawn(function()
			
			if (not transformedTensor) then transformedTensor = self.Function(inputTensorArray) end
			
			self:calculateChainRuleFirstDerivativeTensorArrayAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray)
			
		end)
		
		if (self.FirstDerivativeFunction) then task.spawn(function() self:calculateFirstDerivativeTensorArray(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) end) end
		
	elseif (not self.firstDerivativeFunctionsRequiresTransformedInputTensor) and (self.chainRuleFirstDerivativeFunctionsRequiresTransformedInputTensor) then
		
		task.spawn(function() self:calculateChainRuleFirstDerivativeTensorArrayAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) end)
		
		if (self.FirstDerivativeFunction) then 
			
			task.spawn(function()
				
				if (not transformedTensor) then transformedTensor = self.Function(inputTensorArray) end
				
				self:calculateFirstDerivativeTensorArray(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) 
				
			end) 
			
		end
		
	else
		
		task.spawn(function() self:calculateChainRuleFirstDerivativeTensorArrayAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) end)
		
		if (self.FirstDerivativeFunction) then task.spawn(function() self:calculateFirstDerivativeTensorArray(initialPartialFirstDerivativeTensor, transformedTensor, inputTensorArray) end) end

	end
	
end

function BaseFunctionBlock:differentiate(initialPartialFirstDerivativeTensor) -- Automatic differentiation through reverse accumulation mode. Refer to this Wikipedia article: https://en.wikipedia.org/wiki/Automatic_differentiation

	local numberOfNextFunctionBlocks = #self.NextFunctionBlockArray

	local totalNumberOfDifferentiateFunctionCall = self.totalNumberOfDifferentiateFunctionCall

	local totalInitialPartialFirstDerivativeTensor = self.totalInitialPartialFirstDerivativeTensor

	local waitForAllInitialPartialFirstDerivativeTensors = self.waitForAllInitialPartialFirstDerivativeTensors

	local inputTensorArray = self.inputTensorArray

	local transformedTensor = self.transformedTensor
	
	local transformedTensorDimensionSizeArray = self.transformedTensorDimensionSizeArray

	if (not initialPartialFirstDerivativeTensor) then -- If tensor to be derived is not given, we will use a seed instead.

		initialPartialFirstDerivativeTensor = AqwamTensorLibrary:createTensor(transformedTensorDimensionSizeArray, 1)
		
	else
		
		local initialPartialFirstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(initialPartialFirstDerivativeTensor)
		
		local initialPartialFirstDerivativeTensorNumberOfDimensions = #initialPartialFirstDerivativeTensorDimensionSizeArray
		
		if (initialPartialFirstDerivativeTensorNumberOfDimensions ~= 0) then
			
			local transformedTensorNumberOfDimensions = #transformedTensorDimensionSizeArray

			if (initialPartialFirstDerivativeTensorNumberOfDimensions ~= transformedTensorNumberOfDimensions) then error("Unable to differentiate. The initial partial first derivative tensor has " .. initialPartialFirstDerivativeTensorNumberOfDimensions .. ", but the transformed tensor has " .. transformedTensorNumberOfDimensions .. ".") end

			for dimension, initialPartialFirstDerivativeTensorDimensionSize in ipairs(initialPartialFirstDerivativeTensorDimensionSizeArray)  do

				local transformedTensorDimensionSize = transformedTensorDimensionSizeArray[dimension]

				if (initialPartialFirstDerivativeTensorDimensionSize ~= transformedTensorDimensionSize) then

					error("Unable to differentiate. The initial partial first derivative tensor has a dimension size of " .. initialPartialFirstDerivativeTensorDimensionSize .. " at dimension " .. dimension .. ", but the transformed tensor has " .. transformedTensorDimensionSize .. ".")

				end

			end
			
		end

	end

	if (totalInitialPartialFirstDerivativeTensor) then

		totalInitialPartialFirstDerivativeTensor = AqwamTensorLibrary:add(totalInitialPartialFirstDerivativeTensor, initialPartialFirstDerivativeTensor)

	else

		totalInitialPartialFirstDerivativeTensor = initialPartialFirstDerivativeTensor

	end

	totalNumberOfDifferentiateFunctionCall = totalNumberOfDifferentiateFunctionCall + 1

	self.totalNumberOfDifferentiateFunctionCall = totalNumberOfDifferentiateFunctionCall

	self.totalInitialPartialFirstDerivativeTensor = totalInitialPartialFirstDerivativeTensor

	if (totalNumberOfDifferentiateFunctionCall ~= numberOfNextFunctionBlocks) and (numberOfNextFunctionBlocks ~= 0) and (waitForAllInitialPartialFirstDerivativeTensors) then return nil end

	self.totalNumberOfDifferentiateFunctionCall = 0

	local initialPartialFirstDerivativeTensorToUse

	if (waitForAllInitialPartialFirstDerivativeTensors) then

		initialPartialFirstDerivativeTensorToUse = totalInitialPartialFirstDerivativeTensor

		self.totalInitialPartialFirstDerivativeTensor = nil

	else

		initialPartialFirstDerivativeTensorToUse = initialPartialFirstDerivativeTensor

	end
	
	self:calculateAllFirstDerivativeTensorsAndPassToPreviousFunctionBlocks(initialPartialFirstDerivativeTensorToUse, transformedTensor, inputTensorArray)

end

function BaseFunctionBlock:pullAllInputTensors()

	for i, PreviousFunctionBlock in ipairs(self.PreviousFunctionBlockArray) do

		local transformedTensor = PreviousFunctionBlock:getTransformedTensor(true)

		if (not transformedTensor) then error("Unable to get all the input tensors from previous function blocks.") end

		self:transform(transformedTensor)

	end

end

function BaseFunctionBlock:findNextFunctionBlock(NextFunctionBlockToFind)

	return table.find(self.NextFunctionBlockArray, NextFunctionBlockToFind)

end

function BaseFunctionBlock:findPreviousFunctionBlock(PreviousFunctionBlockToFind)

	return table.find(self.PreviousFunctionBlockArray, PreviousFunctionBlockToFind)

end

function BaseFunctionBlock:addNextFunctionBlock(NextFunctionBlock)

	if (not NextFunctionBlock) then return end

	table.insert(self.NextFunctionBlockArray, NextFunctionBlock)

end

function BaseFunctionBlock:addPreviousFunctionBlock(PreviousFunctionBlock)

	if (not PreviousFunctionBlock) then return end

	table.insert(self.PreviousFunctionBlockArray, PreviousFunctionBlock)

end

function BaseFunctionBlock:addMultipleNextFunctionBlocks(...)

	local NextFunctionBlockArray = {...}

	for i, NextFunctionBlock in ipairs(NextFunctionBlockArray) do

		if self:findNextFunctionBlock(NextFunctionBlock) then continue end

		table.insert(self.NextFunctionBlockArray, NextFunctionBlock)

	end

end

function BaseFunctionBlock:addMultiplePreviousFunctionBlocks(...)

	local PreviousFunctionBlockArray = {...}

	for i, PreviousFunctionBlock in ipairs(PreviousFunctionBlockArray) do

		if self:findPreviousFunctionBlock(PreviousFunctionBlock) then continue end

		table.insert(self.PreviousFunctionBlockArray, PreviousFunctionBlock)

	end

end

function BaseFunctionBlock:linkForward(NextFunctionBlock)

	if (not NextFunctionBlock) then return end

	if (not self:findNextFunctionBlock(NextFunctionBlock)) then self:addNextFunctionBlock(NextFunctionBlock) end

	if (not NextFunctionBlock:findPreviousFunctionBlock(self)) then NextFunctionBlock:addPreviousFunctionBlock(self) end

end

function BaseFunctionBlock:multipleLinkForward(...)

	for i, NextFunctionBlock in ipairs(...) do

		self:linkForward(NextFunctionBlock)

	end

end

function BaseFunctionBlock:linkBackward(PreviousFunctionBlock)

	if (not PreviousFunctionBlock) then return end

	if (not self:findPreviousFunctionBlock(PreviousFunctionBlock)) then self:addPreviousFunctionBlock(PreviousFunctionBlock) end

	if (not PreviousFunctionBlock:findNextFunctionBlock(self)) then PreviousFunctionBlock:addNextFunctionBlock(self) end

end

function BaseFunctionBlock:multipleLinkBackward(...)

	for i, PreviousFunctionBlock in ipairs(...) do

		self:linkBackward(PreviousFunctionBlock)

	end

end

function BaseFunctionBlock:unlinkForward(NextFunctionBlock)

	if (not NextFunctionBlock) then return end

	self:removeNextFunctionBlock(NextFunctionBlock)

	NextFunctionBlock:removePreviousFunctionBlock(self)

end

function BaseFunctionBlock:multipleUnlinkForward(...)

	for i, NextFunctionBlock in ipairs(...) do

		self:unlinkForward(NextFunctionBlock)

	end

end

function BaseFunctionBlock:unlinkBackward(PreviousFunctionBlock)

	if (not PreviousFunctionBlock) then return end

	self:removePreviousFunctionBlock(PreviousFunctionBlock)

	PreviousFunctionBlock:removeNextFunctionBlock(self)

end

function BaseFunctionBlock:multipleUnlinkBackward(...)

	for i, PreviousFunctionBlock in ipairs(...) do

		self:unlinkBackward(PreviousFunctionBlock)

	end

end

function BaseFunctionBlock:setRequiresInputTensors(option)

	self.requiresInputTensors = getValueOrDefaultValue(option, self.requiresInputTensors)

end

function BaseFunctionBlock:getRequiresInputTensors()

	return self.requiresInputTensors

end

function BaseFunctionBlock:setChainRuleFirstDerivativeFunctionRequiresTransformedTensor(option)

	self.chainRuleFirstDerivativeFunctionRequiresTransformedTensor = getValueOrDefaultValue(option, self.chainRuleFirstDerivativeFunctionRequiresTransformedTensor)

end

function BaseFunctionBlock:getChainRuleFirstDerivativeFunctionRequiresTransformedTensor()

	return self.chainRuleFirstDerivativeFunctionRequiresTransformedTensor

end

function BaseFunctionBlock:setFirstDerivativeFunctionRequiresTransformedTensor(option)

	self.firstDerivativeFunctionRequiresTransformedTensor = getValueOrDefaultValue(option, self.firstDerivativeFunctionRequiresTransformedTensor)

end

function BaseFunctionBlock:getFirstDerivativeFunctionRequiresTransformedTensor()

	return self.firstDerivativeFunctionRequiresTransformedTensor

end

function BaseFunctionBlock:setWaitForAllInitialPartialFirstDerivativeTensors(option)

	self.waitForAllInitialPartialFirstDerivativeTensors = getValueOrDefaultValue(option, self.waitForAllInitialPartialFirstDerivativeTensors)

end

function BaseFunctionBlock:getWaitForAllInitialPartialFirstDerivativeTensors(option)

	return self.waitForAllInitialPartialFirstDerivativeTensors

end

function BaseFunctionBlock:setNextFunctionBlockByIndex(nextFunctionBlockArrayIndex, NextFunctionBlock)

	self.NextFunctionBlockArray[nextFunctionBlockArrayIndex] = NextFunctionBlock

end

function BaseFunctionBlock:getNextFunctionBlockByIndex(nextFunctionBlockArrayIndex)

	return self.NextFunctionBlockArray[nextFunctionBlockArrayIndex]

end

function BaseFunctionBlock:setPreviousFunctionBlockByIndex(previousFunctionBlockArrayIndex, PreviousFunctionBlock)

	self.PreviousFunctionBlockArray[previousFunctionBlockArrayIndex] = PreviousFunctionBlock

end

function BaseFunctionBlock:getPreviousFunctionBlockByIndex(previousFunctionBlockArrayIndex)

	return self.PreviousFunctionBlockArray[previousFunctionBlockArrayIndex]

end

function BaseFunctionBlock:removeNextFunctionBlockByIndex(nextFunctionBlockArrayIndex)

	table.remove(self.NextFunctionBlockArray, nextFunctionBlockArrayIndex)

end

function BaseFunctionBlock:removePreviousFunctionBlockByIndex(previousFunctionBlockArrayIndex)

	table.remove(self.PreviousFunctionBlockArray, previousFunctionBlockArrayIndex)

end

function BaseFunctionBlock:removeNextFunctionBlock(NextFunctionBlock)

	local nextFunctionBlockArrayIndex = self:findNextFunctionBlock(NextFunctionBlock)

	if (not nextFunctionBlockArrayIndex) then return end

	table.remove(self.NextFunctionBlockArray, nextFunctionBlockArrayIndex)

end

function BaseFunctionBlock:removePreviousFunctionBlock(PreviousFunctionBlock)

	local previousFunctionBlockArrayIndex = self:findNextFunctionBlock(PreviousFunctionBlock)

	if (not previousFunctionBlockArrayIndex) then return end

	table.remove(self.PreviousFunctionBlockArray, previousFunctionBlockArrayIndex)

end

function BaseFunctionBlock:clearNextFunctionBlockArray()

	table.clear(self.NextFunctionBlockArray)

end

function BaseFunctionBlock:clearPreviousFunctionBlockArray()

	table.clear(self.PreviousFunctionBlockArray)

end

function BaseFunctionBlock:setInputTensorArray(inputTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.inputTensorArray = inputTensorArray

	else

		self.inputTensorArray = deepCopyTable(inputTensorArray)

	end

end

function BaseFunctionBlock:getInputTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.inputTensorArray

	else

		return deepCopyTable(self.inputTensorArray)

	end

end

function BaseFunctionBlock:setTransformedTensor(transformedTensor, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.transformedTensor = transformedTensor

	else

		self.transformedTensor = deepCopyTable(transformedTensor)

	end

end

function BaseFunctionBlock:getTransformedTensor(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.transformedTensor

	else

		return deepCopyTable(self.transformedTensor)

	end

end

function BaseFunctionBlock:setTotalInitialPartialFirstDerivativeTensor(totalInitialPartialFirstDerivativeTensor, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.totalInitialPartialFirstDerivativeTensor = totalInitialPartialFirstDerivativeTensor

	else

		self.totalInitialPartialFirstDerivativeTensor = deepCopyTable(totalInitialPartialFirstDerivativeTensor)

	end

end

function BaseFunctionBlock:getTotalInitialPartialFirstDerivativeTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.totalInitialPartialFirstDerivativeTensor

	else

		return deepCopyTable(self.totalInitialPartialFirstDerivativeTensor)

	end

end

function BaseFunctionBlock:setTotalChainRuleFirstDerivativeTensorArray(totalChainRuleFirstDerivativeTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.totalChainRuleFirstDerivativeTensorArray = totalChainRuleFirstDerivativeTensorArray

	else

		self.totalChainRuleFirstDerivativeTensorArray = deepCopyTable(totalChainRuleFirstDerivativeTensorArray)

	end

end

function BaseFunctionBlock:getTotalChainRuleFirstDerivativeTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.totalChainRuleFirstDerivativeTensorArray

	else

		return deepCopyTable(self.totalChainRuleFirstDerivativeTensorArray)

	end

end

function BaseFunctionBlock:setFirstDerivativeTensorArray(totalFirstDerivativeTensorArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.totalFirstDerivativeTensorArray = totalFirstDerivativeTensorArray

	else

		self.totalFirstDerivativeTensorArray = deepCopyTable(totalFirstDerivativeTensorArray)

	end

end

function BaseFunctionBlock:getTotalFirstDerivativeTensorArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.totalFirstDerivativeTensorArray

	else

		return deepCopyTable(self.totalFirstDerivativeTensorArray)

	end

end

function BaseFunctionBlock:getNextFunctionBlockArray()

	return self.NextFunctionBlockArray

end

function BaseFunctionBlock:getPreviousFunctionBlockArray()

	return self.PreviousFunctionBlockArray

end

function BaseFunctionBlock:setSaveInputTensorArray(option)

	self.saveInputTensorArray = getValueOrDefaultValue(option, self.saveInputTensorArray)

end

function BaseFunctionBlock:getSaveInputTensorArray()

	return self.saveInputTensorArray

end

function BaseFunctionBlock:setSaveTransformedTensor(option)

	self.saveTransformedTensor = getValueOrDefaultValue(option, self.saveTransformedTensor)

end

function BaseFunctionBlock:getSaveTransformedTensor()

	return self.saveTransformedTensor

end

function BaseFunctionBlock:setSaveTotalChainRuleFirstDerivativeTensorArray(option)

	self.saveTotalChainRuleFirstDerivativeTensorArray = getValueOrDefaultValue(option, self.saveTotalChainRuleFirstDerivativeTensorArray)

end

function BaseFunctionBlock:getSaveTotalChainRuleFirstDerivativeTensorArray()

	return self.saveTotalChainRuleFirstDerivativeTensorArray

end

function BaseFunctionBlock:setSaveTotalFirstDerivativeTensorArray(option)

	self.saveTotalFirstDerivativeTensorArray = getValueOrDefaultValue(option, self.saveTotalFirstDerivativeTensorArray)

end

function BaseFunctionBlock:getSaveTotalFirstDerivativeTensorArray()

	return self.saveTotalFirstDerivativeTensorArray

end

function BaseFunctionBlock:waitForInputTensorArray(doNotDeepCopy, waitDuration)

	waitForValue(waitDuration, function()

		return self:getInputTensorArray(true)

	end)

	return self:getInputTensorArray(doNotDeepCopy)

end

function BaseFunctionBlock:waitForTransformedTensor(doNotDeepCopy, waitDuration)

	if (not self.saveTransformedTensor) then

		warn("The setting for saving the transformed tensor is disabled. Returning a nil value.")

		return nil

	end

	waitForValue(waitDuration, function()

		return self:getTransformedTensor(true)

	end)

	return self:getTransformedTensor(doNotDeepCopy)

end

function BaseFunctionBlock:waitForTotalInitialPartialFirstDerivativeTensor(doNotDeepCopy, waitDuration)

	waitForValue(waitDuration, function()

		return self:getTotalInitialPartialFirstDerivativeTensor(true)

	end)

	return self:getTotalInitialPartialFirstDerivativeTensor(doNotDeepCopy)

end

function BaseFunctionBlock:waitForTotalChainRuleFirstDerivativeTensorArray(doNotDeepCopy, waitDuration)

	if (not self.saveTotalChainRuleFirstDerivativeTensorArray) then

		warn("The setting for saving the total chain rule first derivative tensor is disabled. Returning nil values.")

		return {}

	end

	waitForValue(waitDuration, function()

		return self:getTotalChainRuleFirstDerivativeTensorArray(true)

	end)

	return self:getTotalChainRuleFirstDerivativeTensorArray(doNotDeepCopy)

end

function BaseFunctionBlock:waitForTotalFirstDerivativeTensorArray(doNotDeepCopy, waitDuration)

	if (not self.saveTotalFirstDerivativeTensorArray) then

		warn("The setting for saving the total first derivative tensor is disabled. Returning nil values.")

		return {}

	end

	waitForValue(waitDuration, function()

		return self:getTotalFirstDerivativeTensorArray(true)

	end)

	return self:getTotalFirstDerivativeTensorArray(doNotDeepCopy)

end

function BaseFunctionBlock:clearAllStoredTensors()

	self:setInputTensorArray(nil, true)

	self:setTransformedTensor(nil, true)

	self:setTotalInitialPartialFirstDerivativeTensor(nil, true)

	self:setTotalChainRuleFirstDerivativeTensorArray(nil, true)

	self:setFirstDerivativeTensorArray(nil, true)

end

return BaseFunctionBlock