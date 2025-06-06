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

local AHAAutomaticDifferentiatonTensor = {}

--------------------------------------------------------------------------------------

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

local function checkIfIsAutomaticDifferentiationTensor(tensor)

	local isAutomaticDifferentiationTensor = pcall(function()

		tensor:isAutomaticDifferentiationTensor()

	end)

	return isAutomaticDifferentiationTensor

end

local function collapseTensor(tensor, targetDimensionSizeArray)

	local numberOfDimensionsOfTensor = #targetDimensionSizeArray

	local numberOfDimensionsOfDerivativeTensor = #AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensionsToSum = numberOfDimensionsOfDerivativeTensor - numberOfDimensionsOfTensor

	for i = 1, numberOfDimensionsToSum, 1 do tensor = AqwamTensorLibrary:sum(tensor, 1)[1] end

	for i, size in ipairs(targetDimensionSizeArray) do

		if (size == 1) then tensor = AqwamTensorLibrary:sum(tensor, i) end

	end

	return tensor

end

local function createOriginalDimensionArray(targetDimensionArray)

	local originalDimensionArray = {}

	local originalDimension = 1

	for i, targetDimension in ipairs(targetDimensionArray) do

		originalDimensionArray[targetDimension] = originalDimension

		originalDimension = originalDimension + 1

	end

	return originalDimensionArray

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor.new(tensor, PartialDerivativeFunction, previousTensorObject1, previousTensorObject2)

	local self = setmetatable({}, AHAAutomaticDifferentiatonTensor)

	self.tensor = tensor

	self.PartialDerivativeFunction = PartialDerivativeFunction

	self.previousTensorObject1 = previousTensorObject1

	self.previousTensorObject2 = previousTensorObject2

	self.totalDerivativeTensor = nil

	return self

end

function AHAAutomaticDifferentiatonTensor.radian(tensor)
	
	local resultTensor = AqwamTensorLibrary:applyFunction(math.rad, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end
		
		local radiansPerDegree = math.pi / 180

		tensor:differentiate(AqwamTensorLibrary:multiply(radiansPerDegree, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {tensor})
	
end

function AHAAutomaticDifferentiatonTensor.degree(tensor)
	
	local resultTensor = AqwamTensorLibrary:applyFunction(math.deg, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local degreesPerRadian = 180 / math.pi

		tensor:differentiate(AqwamTensorLibrary:multiply(degreesPerRadian, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {tensor})
	
end

function AHAAutomaticDifferentiatonTensor.sin(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.sin, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, tensor)

end

function AHAAutomaticDifferentiatonTensor.cos(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return -math.sin(radian) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, tensor)

end

function AHAAutomaticDifferentiatonTensor.tan(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return math.pow((1 / math.cos(radian)), 2) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, tensor)

end

function AHAAutomaticDifferentiatonTensor.exponent(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.exp, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		tensor:differentiate(AqwamTensorLibrary:multiply(resultTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiatonTensor.logarithm(numberTensor, baseTensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.log, numberTensor, baseTensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(numberTensor) then
			
			local numberTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(numberTensor)

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, numberTensorDimensionSizeArray)

			local partialDerivativeTensor

			if (baseTensor) then

				local partialDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor, baseTensor)

			else

				local partialDerivativeFunctionToApply = function (number) return (1 / number) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor)

			end

			numberTensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

		if checkIfIsAutomaticDifferentiationTensor(baseTensor) then
			
			local baseTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(baseTensor)

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, baseTensorDimensionSizeArray)

			local partialDerivativeFunctionToApply = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, numberTensor, baseTensorDimensionSizeArray)

			baseTensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {numberTensor, baseTensor})

end

function AHAAutomaticDifferentiatonTensor.clamp(tensor, lowerBoundTensor, upperBoundTensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	lowerBoundTensor = lowerBoundTensor or -math.huge

	upperBoundTensor = upperBoundTensor or math.huge

	local resultTensor = AqwamTensorLibrary:applyfunction(math.clamp, tensor, lowerBoundTensor, upperBoundTensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end
			
		local functionToApply = function(value, derivative, lowerBoundValue, upperBoundValue) if ((value >= lowerBoundValue) and (value <= upperBoundValue)) then return value else return 0 end end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor, derivativeTensor, lowerBoundTensor, upperBoundTensor)

		local collapsedPartialDerivativeTensor = collapseTensor(partialDerivativeTensor, dimensionSizeArray)

		tensor:differentiate(collapsedPartialDerivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {tensor, lowerBoundTensor, upperBoundTensor})

end

function AHAAutomaticDifferentiatonTensor.maximum(tensor1, tensor2)
	
	local tensor1DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor1)
	
	local tensor2DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor2)

	tensor1, tensor2 = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor1, tensor2)

	local resultTensor = AqwamTensorLibrary:applyfunction(math.max, tensor1, tensor2)

	local PartialDerivativeFunction = function(derivativeTensor)

		local functionToApply = function(value1, value2)

			if (value2 > value1) then return 2 end

			if (value1 > value2) then return 1 end

			if (value1 == value2) then return 0 end

		end

		local derivativeFunctionfunctionToApply1 = function(index, value)

			if (index == 1) or (index == 0) then return value end

			return 0

		end

		local derivativeFunctionfunctionToApply2 = function(index, value)

			if (index == 2) or (index == 0) then return value end

			return 0

		end

		local indexTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor1, tensor2)

		if checkIfIsAutomaticDifferentiationTensor(tensor1) then

			local partialDerivativeTensor1 = AqwamTensorLibrary:applyFunction(derivativeFunctionfunctionToApply1, indexTensor, derivativeTensor)
			
			local collapsedPartiakDerivativeTensor1 = collapseTensor(partialDerivativeTensor1, tensor1DimensionSizeArray)

			tensor1:differentiate(partialDerivativeTensor1)

		end

		if checkIfIsAutomaticDifferentiationTensor(tensor2) then

			local partialDerivativeTensor2 = AqwamTensorLibrary:applyFunction(derivativeFunctionfunctionToApply2, indexTensor, derivativeTensor)
			
			local collapsedPartiakDerivativeTensor2 = collapseTensor(partialDerivativeTensor2, tensor2DimensionSizeArray)

			tensor2:differentiate(partialDerivativeTensor2)

		end

		--[[

		local functionToApply = function(a) if a then return 1 else return 0 end end

		if checkIfIsAutomaticDifferentiationTensor(tensor1) then
			
			local isGreaterThanBooleanTensor = AqwamTensorLibrary:isGreaterThan(tensor1, tensor2)
			local booleanToNumberTensor = AqwamTensorLibrary:applyFunction(functionToApply, isGreaterThanBooleanTensor)
			local partialDerivativeTensor1 = AqwamTensorLibrary:multiply(booleanToNumberTensor, derivativeTensor)
			
			tensor1:differentiate(partialDerivativeTensor1)
			
		end

		if checkIfIsAutomaticDifferentiationTensor(tensor2) then
			
			local isLessThanOrEqualToBooleanTensor = AqwamTensorLibrary:isLessThanOrEqualTo(tensor1, tensor2)
			local booleanToNumberTensor = AqwamTensorLibrary:applyFunction(functionToApply, isLessThanOrEqualToBooleanTensor)
			local partialDerivativeTensor2 = AqwamTensorLibrary:multiply(booleanToNumberTensor, derivativeTensor)
			
			tensor2:differentiate(partialDerivativeTensor2)
			
		end
		
		--]]

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, tensor1, tensor2)

end

function AHAAutomaticDifferentiatonTensor.minimum(tensor1, tensor2)
	
	local tensor1DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor1)

	local tensor2DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor2)

	tensor1, tensor2 = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensor1, tensor2)

	local resultTensor = AqwamTensorLibrary:applyfunction(math.min, tensor1, tensor2)

	local PartialDerivativeFunction = function(derivativeTensor)

		local functionToApply = function(value1, value2)

			if (value2 < value1) then return 2 end

			if (value1 < value2) then return 1 end

			if (value1 == value2) then return 0 end

		end

		local derivativeFunctionfunctionToApply1 = function(index, value)

			if (index == 0) or (index == 1) then return value end

			return 0

		end

		local derivativeFunctionfunctionToApply2 = function(index, value)

			if (index == 0) or (index == 2) then return value end

			return 0

		end

		local indexTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor1, tensor2)

		if checkIfIsAutomaticDifferentiationTensor(tensor1) then

			local partialDerivativeTensor1 = AqwamTensorLibrary:applyFunction(derivativeFunctionfunctionToApply1, indexTensor, derivativeTensor)
			
			local collapsedPartiakDerivativeTensor1 = collapseTensor(partialDerivativeTensor1, tensor1DimensionSizeArray)

			tensor1:differentiate(collapsedPartiakDerivativeTensor1)

		end

		if checkIfIsAutomaticDifferentiationTensor(tensor2) then

			local partialDerivativeTensor2 = AqwamTensorLibrary:applyFunction(derivativeFunctionfunctionToApply2, indexTensor, derivativeTensor)
			
			local collapsedPartiakDerivativeTensor2 = collapseTensor(partialDerivativeTensor2, tensor2DimensionSizeArray)

			tensor2:differentiate(collapsedPartiakDerivativeTensor2)

		end

		--[[

		local functionToApply = function(a) if a then return 1 else return 0 end end

		if checkIfIsAutomaticDifferentiationTensor(tensor1) then

			local isGreaterThanBooleanTensor = AqwamTensorLibrary:isLessThan(tensor1, tensor2)
			local booleanToNumberTensor = AqwamTensorLibrary:applyFunction(functionToApply, isGreaterThanBooleanTensor)
			local partialDerivativeTensor1 = AqwamTensorLibrary:multiply(booleanToNumberTensor, derivativeTensor)

			tensor1:differentiate(partialDerivativeTensor1)

		end

		if checkIfIsAutomaticDifferentiationTensor(tensor2) then

			local isLessThanOrEqualToBooleanTensor = AqwamTensorLibrary:isGreaterThanOrEqualTo(tensor1, tensor2)
			local booleanToNumberTensor = AqwamTensorLibrary:applyFunction(functionToApply, isLessThanOrEqualToBooleanTensor)
			local partialDerivativeTensor2 = AqwamTensorLibrary:multiply(booleanToNumberTensor, derivativeTensor)

			tensor2:differentiate(partialDerivativeTensor2)

		end
		
		--]]

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, tensor1, tensor2)

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor:__eq(other)

	return AqwamTensorLibrary:isSameTensor(self, other)

end

function AHAAutomaticDifferentiatonTensor:__add(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:add(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(collapsedDerivativeTensor) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(collapsedDerivativeTensor) 

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:__sub(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:subtract(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(collapsedDerivativeTensor) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(collapsedDerivativeTensor) 

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:__mul(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:multiply(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor))

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(AqwamTensorLibrary:multiply(self, collapsedDerivativeTensor))

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:__div(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:divide(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor))

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(AqwamTensorLibrary:multiply(self, collapsedDerivativeTensor))

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:__unm()

	local resultTensor = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return self.new(resultTensor, PartialDerivativeFunction, self)

end

function AHAAutomaticDifferentiatonTensor:__pow(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:power(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor)) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, other)

			other:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	return self.new(resultTensor, PartialDerivativeFunction, self)

end

function AHAAutomaticDifferentiatonTensor:add(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:add(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(collapsedDerivativeTensor) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(collapsedDerivativeTensor) 

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:subtract(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:add(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(collapsedDerivativeTensor) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(collapsedDerivativeTensor) 

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:multiply(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:multiply(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor))

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(AqwamTensorLibrary:multiply(self, collapsedDerivativeTensor))

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:divide(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:divide(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor))

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			other:differentiate(AqwamTensorLibrary:multiply(self, collapsedDerivativeTensor))

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:sum(dimension)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local resultTensor = AqwamTensorLibrary:sum(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 
			
			if (dimension) then
				
				derivativeTensor = AqwamTensorLibrary:expandDimensionSizes(derivativeTensor, dimensionSizeArray)

			else
				
				derivativeTensor = AqwamTensorLibrary:expandNumberOfDimensions(derivativeTensor, dimensionSizeArray)
				
			end
			
			self:differentiate(derivativeTensor) 
			
		end

	end

	return self.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:unaryMinus()

	local resultTensor = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return self.new(resultTensor, PartialDerivativeFunction, self)

end

function AHAAutomaticDifferentiatonTensor:power(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local resultTensor = AqwamTensorLibrary:power(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			self:differentiate(AqwamTensorLibrary:multiply(other, collapsedDerivativeTensor)) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then 

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(function(base, exponent) return (math.pow(base, exponent) * math.log(base)) end, self, other)

			other:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

function AHAAutomaticDifferentiatonTensor:dotProduct(other) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local resultTensor = AqwamTensorLibrary:dotProduct(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		local otherTensor = other:getTensor()

		local otherNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(otherTensor)
		local selfNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(self)

		local transposedOther = AqwamTensorLibrary:transpose(otherTensor, {otherNumberOfDimensions - 1, otherNumberOfDimensions})
		local transposedSelf = AqwamTensorLibrary:transpose(self, {selfNumberOfDimensions - 1, selfNumberOfDimensions})

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:dotProduct(derivativeTensor, transposedOther)) end
		if checkIfIsAutomaticDifferentiationTensor(other) then other:differentiate(AqwamTensorLibrary:dotProduct(transposedSelf, derivativeTensor)) end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, self, other)

	else

		return other.new(resultTensor, PartialDerivativeFunction, self, other)

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor:transpose(dimensionIndexArray)

	local resultTensor = AqwamTensorLibrary:transpose(self, dimensionIndexArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:transpose(derivativeTensor, dimensionIndexArray)) end

	end

	return self.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:flatten(dimensionArray)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local resultTensor = AqwamTensorLibrary:flatten(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, dimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:reshape(dimensionSizeArray)

	local originalDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local resultTensor = AqwamTensorLibrary:reshape(self, dimensionSizeArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, originalDimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:permute(dimensionArray)
	
	local originalDimensionArray = createOriginalDimensionArray(dimensionArray)

	local resultTensor = AqwamTensorLibrary:permute(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:permute(derivativeTensor, originalDimensionArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor:isAutomaticDifferentiationTensor()

	return true

end

function AHAAutomaticDifferentiatonTensor:differentiate(derivativeTensor)

	if (not derivativeTensor) then

		local numberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(self.tensor)

		derivativeTensor = AqwamTensorLibrary:createTensor(table.create(numberOfDimensions, 1), 1)

	end

	local PartialDerivativeFunction = self.PartialDerivativeFunction

	if (PartialDerivativeFunction) then PartialDerivativeFunction(derivativeTensor) end

	local totalDerivativeTensor = self.totalDerivativeTensor

	if (not totalDerivativeTensor) then

		totalDerivativeTensor = derivativeTensor

	else

		totalDerivativeTensor = AqwamTensorLibrary:add(totalDerivativeTensor, derivativeTensor)

	end

	self.totalDerivativeTensor = totalDerivativeTensor 

end

function AHAAutomaticDifferentiatonTensor:copy()

	return deepCopyTable(self)

end

function AHAAutomaticDifferentiatonTensor:getTensor(doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		return self.tensor

	else

		return deepCopyTable(self.tensor)

	end

end

function AHAAutomaticDifferentiatonTensor:setTensor(tensor, doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		self.tensor = tensor

	else

		self.tensor = deepCopyTable(tensor)

	end

end

function AHAAutomaticDifferentiatonTensor:getTotalDerivativeTensor(doNotDeepCopyTable)

	if (doNotDeepCopyTable) then 

		return self.totalDerivativeTensor

	else

		return deepCopyTable(self.totalDerivativeTensor)

	end

end

function AHAAutomaticDifferentiatonTensor:setTotalDerivativeTensor(totalDerivativeTensor, doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		self.totalDerivativeTensor = totalDerivativeTensor

	else

		self.totalDerivativeTensor = deepCopyTable(totalDerivativeTensor)

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor:__tostring()

	return AqwamTensorLibrary:generateTensorString(self.tensor)

end

function AHAAutomaticDifferentiatonTensor:__len()

	return #self.tensor

end

function AHAAutomaticDifferentiatonTensor:__index(index)

	if (type(index) == "number") then

		return rawget(self.tensor, index)

	else

		return rawget(AHAAutomaticDifferentiatonTensor, index)

	end

end

function AHAAutomaticDifferentiatonTensor:__newindex(index, value)

	rawset(self, index, value)

end

function AHAAutomaticDifferentiatonTensor:destroy(areDescendantsDestroyed)

	if (areDescendantsDestroyed) then

		local previousTensorObject1 = self.previousTensorObject1

		local previousTensorObject2 = self.previousTensorObject2

		if checkIfIsAutomaticDifferentiationTensor(previousTensorObject1) then previousTensorObject1:destroy(true) end

		if checkIfIsAutomaticDifferentiationTensor(previousTensorObject2) then previousTensorObject2:destroy(true) end

	end

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AHAAutomaticDifferentiatonTensor
