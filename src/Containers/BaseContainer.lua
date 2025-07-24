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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local BaseContainer = {}

BaseContainer.__index = BaseContainer

setmetatable(BaseContainer, BaseInstance)

local defaultTimeDependent = false

local defaultParallelGradientDescent = true

local function convertToClassTensor(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, ClassesList, cutOffValue)

	local classTensor = {}

	local highestValueTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do classTensor[i], highestValueTensor[i] = convertToClassTensor(tensor[i], dimensionSizeArray, numberOfDimensions, currentDimension + 1, ClassesList, cutOffValue) end

	elseif (dimensionSizeArray[currentDimension] >= 2) then

		local highestValue = -math.huge

		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local value = tensor[i]

			if (value > highestValue) then

				classTensor[1] = ClassesList[i]

				highestValueTensor[1] = value

				highestValue = value

			end

		end

	else

		local value = tensor[1]

		classTensor[1] = ((value >= cutOffValue) and ClassesList[2]) or ClassesList[1]

		highestValueTensor[1] = value

	end

	return classTensor, highestValueTensor

end

function BaseContainer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseContainer = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewBaseContainer, BaseContainer)
	
	NewBaseContainer:setName("BaseContainer")
	
	NewBaseContainer:setClassName("Container")
	
	NewBaseContainer.timeDependent = NewBaseContainer:getValueOrDefaultValue(parameterDictionary.timeDependent, defaultTimeDependent)
	
	NewBaseContainer.parallelGradientDescent = NewBaseContainer:getValueOrDefaultValue(parameterDictionary.parallelGradientDescent, defaultParallelGradientDescent)
	
	NewBaseContainer.InputBlockArray = parameterDictionary.InputBlockArray or {}
	
	NewBaseContainer.WeightBlockArray = parameterDictionary.WeightBlockArray or {}
	
	NewBaseContainer.OutputBlockArray = parameterDictionary.OutputBlockArray or {}
	
	NewBaseContainer.ForwardPropagateFunction = function(...)

		local InputBlockArray = NewBaseContainer.InputBlockArray

		local OutputBlockArray = NewBaseContainer.OutputBlockArray

		local featureTensorArray = {...}

		local outputTensorArray = {}

		for i, OutputBlock in ipairs(OutputBlockArray) do

			OutputBlock:clearAllStoredTensors()

		end

		for i, InputBlock in ipairs(InputBlockArray) do 

			if (not featureTensorArray[i]) then error("No feature tensor for input block " .. i .. ".") end

		end

		for i, InputBlock in ipairs(InputBlockArray) do -- Separate for loops to ensure that we do not run the graph neural network with missing inputs.

			InputBlock:transform(featureTensorArray[i])

		end

		for i, OutputBlock in ipairs(OutputBlockArray) do

			outputTensorArray[i] = OutputBlock:waitForTransformedTensor()

		end
		
		return table.unpack(outputTensorArray)

	end
	
	return NewBaseContainer
	
end

function BaseContainer:setMultipleWeightBlocks(...)

	self.WeightBlockArray = {...}

end

function BaseContainer:setMultipleInputBlocks(...)

	self.InputBlockArray = {...}

end

function BaseContainer:setMultipleOutputBlocks(...)

	self.OutputBlockArray = {...}

end

function BaseContainer:setForwardPropagateFunction(ForwardPropagateFunction)
	
	self.ForwardPropagateFunction = ForwardPropagateFunction
	
end

function BaseContainer:forwardPropagate(...)
	
	return self.ForwardPropagateFunction(...)
	
end

function BaseContainer:setBackwardPropagateFunction(BackwardPropagateFunction)

	self.BackwardPropagateFunction = BackwardPropagateFunction

end

function BaseContainer:backwardPropagate(...)
	
	local lossTensorArray = {...}
	
	local weightLossTensorArray = {}
	
	local WeightBlockArray = self.WeightBlockArray
	
	local OutputBlockArray = self.OutputBlockArray
	
	for i, WeightBlock in ipairs(WeightBlockArray) do 

		WeightBlock:setFirstDerivativeTensorArray(nil, true) -- To ensure that we don't update using old tensor values. Also it is placed here so that we don't accidentally reset the new tensor values stored in function blocks.

	end
	
	for i, OutputBlock in ipairs(OutputBlockArray) do 

		OutputBlock:differentiate(lossTensorArray[i])

	end
	
	for i, WeightBlock in ipairs(WeightBlockArray) do
		
		weightLossTensorArray[i] = WeightBlock:waitForTotalFirstDerivativeTensorArray()[1]
		
	end
	
	return weightLossTensorArray
	
end

function BaseContainer:gradientDescent(weightLossTensorArray)

	for i, WeightBlock in ipairs(self.WeightBlockArray) do

		local weightLossTensor = weightLossTensorArray[i]

		WeightBlock:gradientDescent(weightLossTensor)

	end

end

function BaseContainer:fastUpdate(...)
	
	local lossTensorArray = {...}
	
	local primaryLossTensor = lossTensorArray[1]
	
	local isATensor = (type(primaryLossTensor) == "table")

	local numberOfData = (isATensor and #primaryLossTensor) or 1

	local WeightBlockArray = self.WeightBlockArray
	
	local OutputBlockArray = self.OutputBlockArray

	local numberOfWeightBlocks = #WeightBlockArray
	
	for i, WeightBlock in ipairs(WeightBlockArray) do 
		
		WeightBlock:setFirstDerivativeTensorArray(nil, true) -- To ensure that we don't update using old tensor values. Also it is placed here so that we don't accidentally reset the new tensor values stored in function blocks.
		
	end
	
	for i, OutputBlock in ipairs(OutputBlockArray) do 
		
		OutputBlock:differentiate(lossTensorArray[i])
		
	end
	
	for i = numberOfWeightBlocks, 1, -1 do
		
		task.spawn(function()
			
			local WeightBlock = WeightBlockArray[i]

			local weightLossTensor = WeightBlock:waitForTotalFirstDerivativeTensorArray()[1]

			weightLossTensor = AqwamTensorLibrary:divide(weightLossTensor, numberOfData)

			WeightBlock:gradientDescent(weightLossTensor)

		end)
		
	end

end

function BaseContainer:slowUpdate(...)
	
	local lossTensorArray = {...}

	local primaryLossTensor = lossTensorArray[1]
	
	local isATensor = (type(primaryLossTensor) == "table")

	local numberOfData = (isATensor and #primaryLossTensor) or 1
	
	local backwardPropagateFunction = self.BackwardPropagateFunction
	
	local meanWeightLossTensorArray = {}

	local weightLossTensorArray
	
	if (backwardPropagateFunction) then
		
		weightLossTensorArray = backwardPropagateFunction(...)
		
	else
		
		weightLossTensorArray = self:backwardPropagate(...)
		
	end

	if (numberOfData ~= 1) then 

		for i, weightLossTensor in ipairs(weightLossTensorArray) do

			meanWeightLossTensorArray[i] = AqwamTensorLibrary:divide(weightLossTensor, numberOfData)

		end

	else

		meanWeightLossTensorArray = weightLossTensorArray

	end

	self:gradientDescent(meanWeightLossTensorArray, numberOfData)
	
end

function BaseContainer:update(...)
	
	if ((self.parallelGradientDescent) and (not self.timeDependent)) then
		
		self:fastUpdate(...)
		
	else
		
		self:slowUpdate(...)
		
	end
	
end

function BaseContainer:setConvertToClassTensorFunction(ConvertToClassTensorFunction)
	
	self.ConvertToClassTensorFunction = ConvertToClassTensorFunction
	
end

function BaseContainer:predict(...)

	local parameterArray = {...}

	local numberOfParameters = #parameterArray

	local returnOriginalOutput = false

	if table.find(parameterArray, true) or table.find(parameterArray, false) then 

		returnOriginalOutput = parameterArray[numberOfParameters]

		table.remove(parameterArray, numberOfParameters) 

	end

	local transformedTensorArray = {self:forwardPropagate(table.unpack(parameterArray))}

	if (returnOriginalOutput) then return table.unpack(transformedTensorArray) end

	return self.ConvertToClassTensorFunction(transformedTensorArray)

end

function BaseContainer:setWeightTensorArray(weightTensorArray, doNotDeepCopy)

	weightTensorArray = weightTensorArray or {}

	for i, WeightBlock in ipairs(self.WeightBlockArray) do 

		WeightBlock:setWeightTensor(weightTensorArray[i], doNotDeepCopy)

	end

end

function BaseContainer:getWeightTensorArray(doNotDeepCopy)

	local weightTensorArray = {}

	for i, WeightBlock in ipairs(self.WeightBlockArray) do 

		local weightTensor = WeightBlock:getWeightTensor(doNotDeepCopy)

		table.insert(weightTensorArray, weightTensor)

	end

	return weightTensorArray

end

function BaseContainer:clearAllStoredTensors()

	local ArrayOfBlockArray = {self.InputBlockArray, self.WeightBlockArray, self.OutputBlockArray}

	for i, BlockArray in ipairs(ArrayOfBlockArray) do

		for i, FunctionBlock in ipairs(BlockArray) do 

			FunctionBlock:clearAllStoredTensors()

		end

	end

end

function BaseContainer:getWeightBlockByIndex(index)

	return self.WeightBlockArray[index]

end

function BaseContainer:getWeightBlockArray()

	return self.WeightBlockArray

end

function BaseContainer:getInputBlockByIndex(index)

	return self.InputBlockArray[index]

end

function BaseContainer:getInputBlockArray()

	return self.InputBlockArray

end

function BaseContainer:getOutputBlockByIndex(index)

	return self.OutputBlockArray[index]

end

function BaseContainer:getOutputBlockArray()

	return self.OutputBlockArray

end

function BaseContainer:convertToClassTensor(tensor, ClassesList, cutOffValue)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	if (#ClassesList == 0) then error("No classes.") end
	
	if (not cutOffValue) then error("No cut off value.") end

	return convertToClassTensor(tensor, dimensionSizeArray, #dimensionSizeArray, 1, ClassesList, cutOffValue)

end

return BaseContainer
