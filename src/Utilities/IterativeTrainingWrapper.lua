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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local IterativeTrainingWrapper = {}

IterativeTrainingWrapper.__index = IterativeTrainingWrapper

setmetatable(IterativeTrainingWrapper, BaseInstance)

local defaultMaxNumberOfIterations = 500

local defaultTargetCostValueUpperBound = 0

local defaultTargetCostValueLowerBound = 0

local defaultNumberOfIterationsToCheckIfConverged = math.huge

local defaultNumberOfIterationsPerCostCalculation = 1

local defaultIsOutputPrinted = true

local defaultIterationWaitDuration = nil

local function getValueOrDefaultValue(value, defaultValue)

	if (type(value) == "nil") then return defaultValue end

	return value

end

local function calculateCostValueWhenRequired(currentNumberOfIteration, numberOfIterationsPerCostCalculation, costFunction)

	if ((currentNumberOfIteration % numberOfIterationsPerCostCalculation) == 0) then 

		return costFunction()

	else

		return nil

	end

end

function IterativeTrainingWrapper.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewIterativeTrainingWrapper = BaseInstance.new()
	
	setmetatable(NewIterativeTrainingWrapper, IterativeTrainingWrapper)
	
	NewIterativeTrainingWrapper:setName("IterativeTrainingWrapper")
	
	NewIterativeTrainingWrapper:setClassName("IterativeTrainingWrapper")
	
	NewIterativeTrainingWrapper.Container = nil
	
	NewIterativeTrainingWrapper.maxNumberOfIterations = parameterDictionary.maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewIterativeTrainingWrapper.CostFunctionArray = parameterDictionary.CostFunctionArray or {}
	
	NewIterativeTrainingWrapper.targetCostValueUpperBoundArray = parameterDictionary.targetCostValueUpperBoundArray or {}

	NewIterativeTrainingWrapper.targetCostValueLowerBoundArray = parameterDictionary.targetCostValueLowerBoundArray or {}
	
	NewIterativeTrainingWrapper.numberOfIterationsToCheckIfConvergedArray = parameterDictionary.numberOfIterationsToCheckIfConvergedArray or {}
	
	NewIterativeTrainingWrapper.numberOfIterationsPerCostCalculation = parameterDictionary.numberOfIterationsPerCostCalculation or defaultNumberOfIterationsPerCostCalculation
	
	NewIterativeTrainingWrapper.isOutputPrinted = getValueOrDefaultValue(parameterDictionary.isOutputPrinted, defaultIsOutputPrinted)
	
	NewIterativeTrainingWrapper.iterationWaitDuration = parameterDictionary.iterationWaitDuration or defaultIterationWaitDuration

	NewIterativeTrainingWrapper.currentCostToCheckForConvergenceArray = {}

	NewIterativeTrainingWrapper.currentNumberOfIterationsToCheckIfConvergedArray = {}
	
	return NewIterativeTrainingWrapper
	
end

function IterativeTrainingWrapper:setParameters(parameterDictionary)
	
	self.maxNumberOfIterations = parameterDictionary.maxNumberOfIterations or self.maxNumberOfIterations

	self.CostFunctionArray = parameterDictionary.CostFunctionArray or self.CostFunctionArray

	self.targetCostValueUpperBoundArray = parameterDictionary.targetCostValueUpperBoundArray or self.targetCostValueUpperBoundArray

	self.targetCostValueLowerBoundArray = parameterDictionary.targetCostValueLowerBoundArray or self.targetCostValueLowerBoundArray
	
	self.numberOfIterationsToCheckIfConvergedArray = parameterDictionary.numberOfIterationsToCheckIfConvergedArray or self.numberOfIterationsToCheckIfConvergedArray
	
	self.numberOfIterationsPerCostCalculation = parameterDictionary.numberOfIterationsPerCostCalculation or self.numberOfIterationsPerCostCalculation
	
	self.isOutputPrinted = getValueOrDefaultValue(parameterDictionary.isOutputPrinted, self.isOutputPrinted)
	
	self.iterationWaitDuration = parameterDictionary.iterationWaitDuration
	
end

function IterativeTrainingWrapper:setContainer(Container)
	
	self.Container = Container
	
end

function IterativeTrainingWrapper:getContainer(Container)
	
	return self.Container
	
end

function IterativeTrainingWrapper:checkIfConverged(index, cost)

	if (not cost) then return false end
	
	local hasConverged
	
	local numberOfIterationsToCheckIfConverged = self.numberOfIterationsToCheckIfConvergedArray[index] or defaultNumberOfIterationsToCheckIfConverged
	
	local currentNumberOfIterationsToCheckIfConverged = self.currentNumberOfIterationsToCheckIfConvergedArray[index]
	
	local currentCostToCheckForConvergence = self.currentCostToCheckForConvergenceArray[index]

	if (not currentCostToCheckForConvergence) then

		currentCostToCheckForConvergence = cost

		hasConverged = false

	elseif (currentCostToCheckForConvergence ~= cost) then

		currentNumberOfIterationsToCheckIfConverged = 1

		currentCostToCheckForConvergence = cost

		hasConverged = false

	elseif (currentNumberOfIterationsToCheckIfConverged < numberOfIterationsToCheckIfConverged) then

		currentNumberOfIterationsToCheckIfConverged = currentNumberOfIterationsToCheckIfConverged + 1

		hasConverged = false
		
	else
		
		hasConverged = true

	end
	
	self.numberOfIterationsToCheckIfConvergedArray[index] = numberOfIterationsToCheckIfConverged

	self.currentNumberOfIterationsToCheckIfConvergedArray[index] = currentNumberOfIterationsToCheckIfConverged

	self.currentCostToCheckForConvergenceArray[index] = numberOfIterationsToCheckIfConverged
	
	return hasConverged

end

function IterativeTrainingWrapper:checkIfTargetCostValueReached(index, costValue)

	if (not costValue) then return false end
	
	local targetCostValueLowerBound = self.targetCostValueLowerBoundArray[index] or defaultTargetCostValueLowerBound
	
	local targetCostValueUpperBound = self.targetCostValueUpperBoundArray[index] or defaultTargetCostValueUpperBound

	return (costValue >= targetCostValueLowerBound) and (costValue <= targetCostValueUpperBound)

end

function IterativeTrainingWrapper:train(featureTensorArray, labelTensorArray)
	
	local currentNumberOfIteration = 0
	
	local Container = self.Container
	
	local CostFunctionArray = self.CostFunctionArray

	local maxNumberOfIterations = self.maxNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local targetCostValueUpperBoundArray = self.targetCostValueUpperBoundArray
	
	local targetCostValueLowerBoundArray = self.targetCostValueLowerBoundArray
	
	local numberOfIterationsPerCostCalculation = self.numberOfIterationsPerCostCalculation
	
	local iterationWaitDuration =  self.iterationWaitDuration
	
	local arrayOfCostValueArray = {}
	
	local stopConverging = false
	
	if (not Container) then error("No model!") end
	
	if (#CostFunctionArray == 0) then error("No cost functions!") end
	
	repeat
		
		if (type(iterationWaitDuration) ~= "nil") then
			
			if (type(iterationWaitDuration) == "number") then
				
				task.wait(iterationWaitDuration)
				
			else
				
				task.wait()
				
			end
			
		end
		
		local lossTensorArray = {}

		local costValueArray = {}
		
		currentNumberOfIteration = currentNumberOfIteration + 1
		
		local outputTensorArray = {Container:forwardPropagate(table.unpack(featureTensorArray))}
		
		for i, CostFunction in ipairs(CostFunctionArray) do
			
			local outputTensor = outputTensorArray[i]
			
			local labelTensor = labelTensorArray[i]
			
			local lossTensor = CostFunction:calculateLossTensor(outputTensor, labelTensor)
			
			table.insert(lossTensorArray, lossTensor)
			
			local costValue = calculateCostValueWhenRequired(currentNumberOfIteration, numberOfIterationsPerCostCalculation, function()

				return CostFunction:calculateCostValue(outputTensor, labelTensor)

			end)
			
			if (costValue) then table.insert(costValueArray, costValue) end
			
		end
		
		Container:backwardPropagate(table.unpack(lossTensorArray))
		
		if (#costValueArray <= 0) then continue end
			
		table.insert(arrayOfCostValueArray, costValueArray) 
			
		if (isOutputPrinted) then
			
			local stringToBePrinted = "\n\n====================\nIteration: " .. currentNumberOfIteration .. " \n====================\n"

			for i, cost in ipairs(costValueArray) do

				stringToBePrinted = stringToBePrinted .. "\nCost " .. i .. ": " .. cost

			end

			stringToBePrinted = stringToBePrinted .. "\n\n"

			print(stringToBePrinted) 
			
		end
		
		for i, costValue in ipairs(costValueArray) do
			
			if (not self:checkIfConverged(i, costValue)) and (not self:checkIfTargetCostValueReached(i, costValue)) then continue end
			
			stopConverging = true
			
			break
			
		end
				
	until (currentNumberOfIteration >= maxNumberOfIterations) or (stopConverging)
	
	return arrayOfCostValueArray
	
end

return IterativeTrainingWrapper