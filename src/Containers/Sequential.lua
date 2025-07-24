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

local BaseContainer = require(script.Parent.BaseContainer)

local SequentialContainer = {}

SequentialContainer.__index = SequentialContainer

setmetatable(SequentialContainer, BaseContainer)

local defaultCutOffValue = 0

function SequentialContainer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewSequentialContainer = BaseContainer.new(parameterDictionary)

	setmetatable(NewSequentialContainer, SequentialContainer)

	NewSequentialContainer:setName("Sequential")

	NewSequentialContainer.FunctionBlockArray = parameterDictionary.FunctionBlockArray or {}
	
	NewSequentialContainer.ClassesList = parameterDictionary.ClassesList or {}
	
	NewSequentialContainer.cutOffValue = parameterDictionary.cutOffValue or defaultCutOffValue
	
	NewSequentialContainer:setConvertToClassTensorFunction(function(transformedTensorArray)
		
		return NewSequentialContainer:convertToClassTensor(transformedTensorArray[1], NewSequentialContainer.ClassesList, NewSequentialContainer.cutOffValue)
		
	end)

	return NewSequentialContainer

end

function SequentialContainer:setMultipleFunctionBlocks(...)

	local FunctionBlockArray = {...}
	
	local numberOfFunctionBlocks = #FunctionBlockArray
	
	local WeightBlockArray = {}
	
	for i, FunctionBlock in ipairs(FunctionBlockArray) do
		
		local functionBlockClassName = FunctionBlock:getClassName()

		local CurrentFunctionBlock = FunctionBlockArray[i]

		local NextFunctionBlock = FunctionBlockArray[i + 1]

		CurrentFunctionBlock:linkForward(NextFunctionBlock)

		if (functionBlockClassName == "WeightBlock") or (functionBlockClassName == "ConvolutionBlock") then

			table.insert(WeightBlockArray, FunctionBlock)			
			
		end
		
	end
	
	self.InputBlockArray = {FunctionBlockArray[1]}
	
	self.WeightBlockArray = WeightBlockArray
	
	self.OutputBlockArray = {FunctionBlockArray[numberOfFunctionBlocks]}
	
	self.FunctionBlockArray = FunctionBlockArray

end

function SequentialContainer:setCutOffValue(cutOffValue)

	self.cutOffValue = cutOffValue or self.cutOffValue

end

function SequentialContainer:getCutOffValue()

	return self.cutOffValue

end

function SequentialContainer:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function SequentialContainer:getClassesList()

	return self.ClassesList

end

function SequentialContainer:detachAllFunctionBlocks()

	local FunctionBlockArray = self.FunctionBlockArray

	for i = 1, #FunctionBlockArray, 1 do

		local CurrentFunctionBlock = FunctionBlockArray[i]

		local NextFunctionBlock = FunctionBlockArray[i + 1]

		CurrentFunctionBlock:unlinkForward(NextFunctionBlock)

	end

end

function SequentialContainer:getFunctionBlockByIndex(index)

	return self.FunctionBlockArray[index]

end

function SequentialContainer:getFunctionBlockArray()

	return self.FunctionBlockArray

end

return SequentialContainer