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

local GatedRecurrentUnitCell = require(script.Parent.GatedRecurrentUnitCell)

local GatedRecurrentUnitContainer = {}

GatedRecurrentUnitContainer.__index = GatedRecurrentUnitContainer

setmetatable(GatedRecurrentUnitContainer, BaseContainer)

local defaultReverse = false

function GatedRecurrentUnitContainer.new(parameterDictionary)

	local NewGatedRecurrentUnitContainer = BaseContainer.new()

	setmetatable(NewGatedRecurrentUnitContainer, GatedRecurrentUnitContainer)

	NewGatedRecurrentUnitContainer:setName("GatedRecurrentUnit")

	parameterDictionary = parameterDictionary or {}

	NewGatedRecurrentUnitContainer.GatedRecurrentUnitCell = parameterDictionary.GatedRecurrentUnitCell or GatedRecurrentUnitCell.new(parameterDictionary)
	
	NewGatedRecurrentUnitContainer.reverse = NewGatedRecurrentUnitContainer:getValueOrDefaultValue(parameterDictionary.reverse, defaultReverse)
	
	NewGatedRecurrentUnitContainer.timeDependent = true
	
	NewGatedRecurrentUnitContainer.hiddenStateTensor = parameterDictionary.hiddenStateTensor
	
	NewGatedRecurrentUnitContainer.transformedTensor = parameterDictionary.transformedTensor
	
	NewGatedRecurrentUnitContainer:setForwardPropagateFunction(function(featureTensor)
		
		local GatedRecurrentUnitCell = NewGatedRecurrentUnitContainer.GatedRecurrentUnitCell

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureTensor)

		local numberOfDimensions = #dimensionSizeArray

		local numberOfData = dimensionSizeArray[1]

		local numberOfTimeSteps = dimensionSizeArray[2]

		local hiddenStateDimensionSizeArray = {numberOfData, 1, GatedRecurrentUnitCell.hiddenDimensionSize}

		local transformedSubTensor = AqwamTensorLibrary:createTensor(hiddenStateDimensionSizeArray, 0)

		local transformedTensor
		
		if (NewGatedRecurrentUnitContainer.reverse) then
			
			for timeStep = numberOfTimeSteps, 1, -1 do

				local originDimensionIndexArray = table.create(numberOfDimensions, 1)

				local targetDimensionIndexArray = table.clone(dimensionSizeArray)

				originDimensionIndexArray[2] = timeStep

				targetDimensionIndexArray[2] = timeStep

				local featureSubTensor = AqwamTensorLibrary:extract(featureTensor, originDimensionIndexArray, targetDimensionIndexArray)

				transformedSubTensor = GatedRecurrentUnitCell:forwardPropagate(featureSubTensor, transformedSubTensor)

				if (transformedTensor) then

					transformedTensor = AqwamTensorLibrary:concatenate(transformedSubTensor, transformedTensor, 2)

				else

					transformedTensor = transformedSubTensor 

				end

			end
			
		else
			
			for timeStep = 1, numberOfTimeSteps, 1 do

				local originDimensionIndexArray = table.create(numberOfDimensions, 1)

				local targetDimensionIndexArray = table.clone(dimensionSizeArray)

				originDimensionIndexArray[2] = timeStep

				targetDimensionIndexArray[2] = timeStep

				local featureSubTensor = AqwamTensorLibrary:extract(featureTensor, originDimensionIndexArray, targetDimensionIndexArray)

				transformedSubTensor = GatedRecurrentUnitCell:forwardPropagate(featureSubTensor, transformedSubTensor)

				if (transformedTensor) then

					transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, transformedSubTensor, 2)

				else

					transformedTensor = transformedSubTensor 

				end

			end
			
		end

		NewGatedRecurrentUnitContainer.featureTensor = featureTensor

		NewGatedRecurrentUnitContainer.transformedTensor = transformedTensor

		return transformedTensor
		
	end)
	
	NewGatedRecurrentUnitContainer:setBackwardPropagateFunction(function(lossTensor)
		
		local GatedRecurrentUnitCell = NewGatedRecurrentUnitContainer.GatedRecurrentUnitCell
		
		local reverse = NewGatedRecurrentUnitContainer.reverse

		local featureTensor = NewGatedRecurrentUnitContainer.featureTensor

		local transformedTensor = NewGatedRecurrentUnitContainer.transformedTensor

		local GatedRecurrentUnitCell = NewGatedRecurrentUnitContainer.GatedRecurrentUnitCell

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureTensor)

		local numberOfDimensions = #dimensionSizeArray

		local numberOfData = dimensionSizeArray[1]

		local numberOfTimeSteps = dimensionSizeArray[2]

		local hiddenStateDimensionSizeArray = {numberOfData, 1, GatedRecurrentUnitCell.hiddenDimensionSize}
		
		local startingTimeStep = (reverse and 1) or numberOfTimeSteps
		
		local finalTimeStep = (reverse and numberOfTimeSteps) or 1
		
		local timeStepChange = (reverse and 1) or -1
		
		local weightLossTensorArray = {}

		local originDimensionIndexArray

		local targetDimensionIndexArray

		local transformedTensorDimensionIndexArray

		local featureSubTensor

		local extractedLossTensor

		local transformedSubTensor

		local weightLossSubTensorArray
		
		for timeStep = startingTimeStep, finalTimeStep, timeStepChange do

			originDimensionIndexArray = table.create(numberOfDimensions, 1)

			targetDimensionIndexArray = table.clone(dimensionSizeArray)

			originDimensionIndexArray[2] = timeStep

			targetDimensionIndexArray[2] = timeStep

			featureSubTensor = AqwamTensorLibrary:extract(featureTensor, originDimensionIndexArray, targetDimensionIndexArray)

			extractedLossTensor = AqwamTensorLibrary:extract(lossTensor, originDimensionIndexArray, targetDimensionIndexArray)
				
			if (reverse and (timeStep == numberOfTimeSteps)) or ((not reverse) and (timeStep == 1)) then
				
				transformedSubTensor = AqwamTensorLibrary:createTensor(hiddenStateDimensionSizeArray, 0)
				
			else
				
				transformedSubTensor = AqwamTensorLibrary:extract(transformedTensor, originDimensionIndexArray, targetDimensionIndexArray)
				
			end
			
			GatedRecurrentUnitCell:forwardPropagate(featureSubTensor, transformedSubTensor)

			weightLossSubTensorArray = GatedRecurrentUnitCell:backwardPropagate(extractedLossTensor)

			for i, weightLossSubTensor in ipairs(weightLossSubTensorArray) do

				local weightLossTensor = weightLossTensorArray[i]

				if (weightLossTensor) then

					weightLossTensor = AqwamTensorLibrary:add(weightLossTensor, weightLossSubTensor)

				else

					weightLossTensor = weightLossSubTensor

				end

				weightLossTensorArray[i] = weightLossTensor

			end

		end

		return weightLossTensorArray
		
	end)

	return NewGatedRecurrentUnitContainer

end

function GatedRecurrentUnitContainer:gradientDescent(weightLossTensorArray, numberOfData)

	self.GatedRecurrentUnitCell:gradientDescent(weightLossTensorArray, numberOfData)

end

function GatedRecurrentUnitContainer:clearAllStoredTensors()
	
	self.GatedRecurrentUnitCell:clearAllStoredTensors()
	
end

function GatedRecurrentUnitContainer:setWeightTensorArray(weightTensorArray, doNotDeepCopy)
	
	self.GatedRecurrentUnitCell:setWeightTensorArray(weightTensorArray, doNotDeepCopy)

end

function GatedRecurrentUnitContainer:getWeightTensorArray(doNotDeepCopy)

	return self.GatedRecurrentUnitCell:getWeightTensorArray(doNotDeepCopy)

end

return GatedRecurrentUnitContainer