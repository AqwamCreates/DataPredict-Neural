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

local ComputationalGraphContainer = {}

ComputationalGraphContainer.__index = ComputationalGraphContainer

setmetatable(ComputationalGraphContainer, BaseContainer)

function ComputationalGraphContainer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewComputationalGraphContainer = BaseContainer.new(parameterDictionary)

	setmetatable(NewComputationalGraphContainer, ComputationalGraphContainer)

	NewComputationalGraphContainer:setName("ComputationalGraph")
	
	NewComputationalGraphContainer.ClassesListArray = parameterDictionary.ClassesListArray or {}
	
	NewComputationalGraphContainer.cutOffValueArray = parameterDictionary.cutOffValueArray or {}
	
	NewComputationalGraphContainer:setConvertToClassTensorFunction(function(transformedTensorArray)
		
		local ClassesListArray = NewComputationalGraphContainer.ClassesListArray
		
		local cutOffValueArray = NewComputationalGraphContainer.cutOffValueArray
		
		local classTensorArray = {}
		
		for i, transformedTensor  in ipairs(transformedTensorArray) do
			
			classTensorArray[i] = NewComputationalGraphContainer:convertToClassTensor(transformedTensor, ClassesListArray[i] or {}, cutOffValueArray[i] or 0)
			
		end

		return table.unpack(classTensorArray)

	end)

	return NewComputationalGraphContainer

end

return ComputationalGraphContainer