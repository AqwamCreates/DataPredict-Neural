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

local TensorToClassConverter = {}

TensorToClassConverter.__index = TensorToClassConverter

setmetatable(TensorToClassConverter, BaseInstance)

local defaultCutOffValue = 0

function TensorToClassConverter.new(parameterDictionary)
	
	local NewTensorToClassConverter = BaseInstance.new()
	
	setmetatable(NewTensorToClassConverter, TensorToClassConverter)
	
	NewTensorToClassConverter:setName("TensorToClassConverter")
	
	NewTensorToClassConverter:setClassName("TensorToClassConverter")
	
	parameterDictionary = parameterDictionary or {}
	
	NewTensorToClassConverter.ClassesList = parameterDictionary.ClassesList or {}
	
	NewTensorToClassConverter.cutOffValue = parameterDictionary.cutOffValue or defaultCutOffValue
	
	return NewTensorToClassConverter
	
end

function TensorToClassConverter:setCutOffValue(cutOffValue)

	self.cutOffValue = cutOffValue or self.cutOffValue

end

function TensorToClassConverter:getCutOffValue()

	return self.setCutOffValue

end

function TensorToClassConverter:setClassesList(ClassesList)
	
	self.ClassesList = ClassesList or self.ClassesList
	
end

function TensorToClassConverter:getClassesList()

	return self.ClassesList

end

local function convert(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, ClassesList, cutOffValue)
	
	local classTensor = {}
	
	if (currentDimension < numberOfDimensions) then
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do classTensor[i] = convert(tensor[i], dimensionSizeArray, numberOfDimensions, currentDimension + 1, ClassesList, cutOffValue) end
		
	elseif (dimensionSizeArray[currentDimension] >= 2) then
		
		local highestValue = -math.huge
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do

			local value = tensor[i]

			if (value > highestValue) then
				
				classTensor[1] = ClassesList[i]

				highestValue = value
				
			end
			
		end
		
	else
		
		classTensor[1] = ((tensor[1] >= cutOffValue) and ClassesList[2]) or ClassesList[1]
		
	end
	
	return classTensor
	
end

function TensorToClassConverter:convert(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
	
	return convert(tensor, dimensionSizeArray, #dimensionSizeArray, 1, self.ClassesList, self.cutOffValue)
	
end

return TensorToClassConverter