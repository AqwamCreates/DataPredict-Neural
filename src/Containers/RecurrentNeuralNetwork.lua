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

local RecurrentNeuralNetwork = require(script.Parent.RecurrentNeuralNetworkCell)

local RecurrentNeuralNetworkContainer = {}

RecurrentNeuralNetworkContainer.__index = RecurrentNeuralNetworkContainer

setmetatable(RecurrentNeuralNetworkContainer, BaseContainer)

local defaultReverse = false

function RecurrentNeuralNetworkContainer.new(parameterDictionary)

	local NewRecurrentNeuralNetworkContainer = BaseContainer.new()

	setmetatable(NewRecurrentNeuralNetworkContainer, RecurrentNeuralNetworkContainer)

	NewRecurrentNeuralNetworkContainer:setName("RecurrentNeuralNetwork")

	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.timeDependent = true

	NewRecurrentNeuralNetworkContainer.RecurrentNeuralNetworkCell = parameterDictionary.RecurrentNeuralNetworkCell or RecurrentNeuralNetwork.new(parameterDictionary)
	
	NewRecurrentNeuralNetworkContainer.reverse = NewRecurrentNeuralNetworkContainer:getValueOrDefaultValue(parameterDictionary.reverse, defaultReverse)
	
	NewRecurrentNeuralNetworkContainer.timeDependent = true
	
	NewRecurrentNeuralNetworkContainer.hiddenStateTensor = parameterDictionary.hiddenStateTensor
	
	NewRecurrentNeuralNetworkContainer.transformedTensor = parameterDictionary.transformedTensor
	
	NewRecurrentNeuralNetworkContainer:setForwardPropagateFunction(function(featureTensor)
		
		local RecurrentNeuralNetworkCell = NewRecurrentNeuralNetworkContainer.RecurrentNeuralNetworkCell

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureTensor)

		local numberOfDimensions = #dimensionSizeArray

		local numberOfData = dimensionSizeArray[1]

		local numberOfTimeSteps = dimensionSizeArray[2]

		local hiddenStateDimensionSizeArray = {numberOfData, 1, RecurrentNeuralNetworkCell.hiddenDimensionSize}

		local transformedSubTensor = AqwamTensorLibrary:createTensor(hiddenStateDimensionSizeArray, 0)

		local transformedTensor
		
		if (NewRecurrentNeuralNetworkContainer.reverse) then
			
			for timeStep = numberOfTimeSteps, 1, -1 do

				local originDimensionIndexArray = table.create(numberOfDimensions, 1)

				local targetDimensionIndexArray = table.clone(dimensionSizeArray)

				originDimensionIndexArray[2] = timeStep

				targetDimensionIndexArray[2] = timeStep

				local featureSubTensor = AqwamTensorLibrary:extract(featureTensor, originDimensionIndexArray, targetDimensionIndexArray)

				transformedSubTensor = RecurrentNeuralNetworkCell:forwardPropagate(featureSubTensor, transformedSubTensor)

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

				transformedSubTensor = RecurrentNeuralNetworkCell:forwardPropagate(featureSubTensor, transformedSubTensor)

				if (transformedTensor) then

					transformedTensor = AqwamTensorLibrary:concatenate(transformedTensor, transformedSubTensor, 2)

				else

					transformedTensor = transformedSubTensor 

				end

			end
			
		end

		NewRecurrentNeuralNetworkContainer.featureTensor = featureTensor

		NewRecurrentNeuralNetworkContainer.transformedTensor = transformedTensor

		return transformedTensor
		
	end)
	
	NewRecurrentNeuralNetworkContainer:setBackwardPropagateFunction(function(lossTensor) 
		
		local RecurrentNeuralNetworkCell = NewRecurrentNeuralNetworkContainer.RecurrentNeuralNetworkCell
		
		local reverse = NewRecurrentNeuralNetworkContainer.reverse

		local featureTensor = NewRecurrentNeuralNetworkContainer.featureTensor

		local transformedTensor = NewRecurrentNeuralNetworkContainer.transformedTensor

		local RecurrentNeuralNetworkCell = NewRecurrentNeuralNetworkContainer.RecurrentNeuralNetworkCell

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureTensor)

		local numberOfDimensions = #dimensionSizeArray

		local numberOfData = dimensionSizeArray[1]

		local numberOfTimeSteps = dimensionSizeArray[2]

		local hiddenStateDimensionSizeArray = {numberOfData, 1, RecurrentNeuralNetworkCell.hiddenDimensionSize}
		
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
			
			RecurrentNeuralNetworkCell:forwardPropagate(featureSubTensor, transformedSubTensor)

			weightLossSubTensorArray = RecurrentNeuralNetworkCell:backwardPropagate(extractedLossTensor)

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

	return NewRecurrentNeuralNetworkContainer

end

function RecurrentNeuralNetworkContainer:gradientDescent(weightLossTensorArray, numberOfData)

	self.RecurrentNeuralNetworkCell:gradientDescent(weightLossTensorArray, numberOfData)

end

function RecurrentNeuralNetworkContainer:clearAllStoredTensors()
	
	self.RecurrentNeuralNetworkCell:clearAllStoredTensors()
	
end

function RecurrentNeuralNetworkContainer:setWeightTensorArray(weightTensorArray, doNotDeepCopy)
	
	self.RecurrentNeuralNetworkCell:setWeightTensorArray(weightTensorArray, doNotDeepCopy)

end

function RecurrentNeuralNetworkContainer:getWeightTensorArray(doNotDeepCopy)

	return self.RecurrentNeuralNetworkCell:getWeightTensorArray(doNotDeepCopy)

end

return RecurrentNeuralNetworkContainer