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

function AHAAutomaticDifferentiatonTensor.new(tensor, PartialDerivativeFunction, tensorArray)

	local self = setmetatable({}, AHAAutomaticDifferentiatonTensor)

	self.tensor = tensor

	self.PartialDerivativeFunction = PartialDerivativeFunction

	self.tensorArray = tensorArray

	self.totalDerivativeTensor = nil

	return self

end

function AHAAutomaticDifferentiatonTensor.radian(tensor)
	
	local result = AqwamTensorLibrary:applyFunction(math.rad, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end
		
		local radiansPerDegree = math.pi / 180

		tensor:differentiate(AqwamTensorLibrary:multiply(radiansPerDegree, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor})
	
end

function AHAAutomaticDifferentiatonTensor.degree(tensor)
	
	local result = AqwamTensorLibrary:applyFunction(math.deg, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local degreesPerRadian = 180 / math.pi

		tensor:differentiate(AqwamTensorLibrary:multiply(degreesPerRadian, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor})
	
end

function AHAAutomaticDifferentiatonTensor.sin(tensor)

	local result = AqwamTensorLibrary:applyFunction(math.sin, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiatonTensor.cos(tensor)

	local result = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return -math.sin(radian) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiatonTensor.tan(tensor)

	local result = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return math.pow((1 / math.cos(radian)), 2) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiatonTensor.exponent(tensor)

	local result = AqwamTensorLibrary:applyFunction(math.exp, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		tensor:differentiate(AqwamTensorLibrary:multiply(result, derivativeTensor))

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiatonTensor.clamp(tensor, lowerBoundTensor, upperBoundTensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	lowerBoundTensor = lowerBoundTensor or -math.huge

	upperBoundTensor = upperBoundTensor or math.huge

	local result = AqwamTensorLibrary:applyfunction(math.clamp, tensor, lowerBoundTensor, upperBoundTensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end
			
		local functionToApply = function(value, derivative, lowerBoundValue, upperBoundValue) if ((value >= lowerBoundValue) and (value <= upperBoundValue)) then return value else return 0 end end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, tensor, derivativeTensor, lowerBoundTensor, upperBoundTensor)

		local collapsedPartialDerivativeTensor = collapseTensor(partialDerivativeTensor, dimensionSizeArray)

		tensor:differentiate(collapsedPartialDerivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {tensor, lowerBoundTensor, upperBoundTensor})

end

function AHAAutomaticDifferentiatonTensor.maximum(...)
	
	local tensorArray = {...}
	
	local numberOfTensors = #tensorArray
	
	local dimensionSizeArrayArray = {}
	
	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])
	
	for i = 1, (numberOfTensors - 1), 1 do
		
		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i + 1])
		
		expandedTensorArray[i], expandedTensorArray[i + 1] = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensorArray[i], tensorArray[i + 1])
		
	end
	
	local result = AqwamTensorLibrary:applyfunction(math.max, ...)

	local PartialDerivativeFunction = function(derivativeTensor)
		
		for i, tensor in ipairs(tensorArray) do
			
			if checkIfIsAutomaticDifferentiationTensor(tensor) then
				
				local functionToApply = function(derivativeValue, ...)

					local isMaximum = false

					local highestValue = -math.huge

					for j, value in ipairs(...) do

						if (value >= highestValue) then

							isMaximum = (i == j)

							highestValue = value

						end

					end

					return (isMaximum and derivativeValue) or 0

				end
				
				local currentDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, derivativeTensor, table.unpack(tensorArray))

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate(collapsedCurrentDerivativeTensor) 

			end
			
		end

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, tensorArray)

end

function AHAAutomaticDifferentiatonTensor.minimum(...)
	
	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 1, (numberOfTensors - 1), 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i + 1])

		expandedTensorArray[i], expandedTensorArray[i + 1] = AqwamTensorLibrary:broadcastATensorIfDifferentSize(tensorArray[i], tensorArray[i + 1])

	end

	local result = AqwamTensorLibrary:applyfunction(math.min, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local functionToApply = function(derivativeValue, ...)

					local isMinimum = false

					local lowestValue = -math.huge

					for j, value in ipairs(...) do

						if (value <= lowestValue) then

							isMinimum = (i == j)

							lowestValue = value

						end

					end

					return (isMinimum and derivativeValue) or 0

				end

				local currentDerivativeTensor = AqwamTensorLibrary:applyFunction(functionToApply, derivativeTensor, table.unpack(tensorArray))

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate(collapsedCurrentDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, tensorArray)

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor:__eq(other)

	return AqwamTensorLibrary:isSameTensor(self, other)

end

function AHAAutomaticDifferentiatonTensor:__add(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:add(self, other)

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

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:__sub(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:subtract(self, other)

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

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:__mul(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:multiply(self, other)

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

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:__div(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:divide(self, other)

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

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:__unm()

	local result = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return self.new(result, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:__pow(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:power(self, other)

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

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:add(...)

	local tensorArray = {self, ...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	for i, tensor in ipairs(tensorArray) do
		
		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensor)
		
	end

	local result = AqwamTensorLibrary:add(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then
				
				local collapsedDerivativeTensor = collapseTensor(derivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate(collapsedDerivativeTensor) 
				
			end
			
		end

	end
	
	for _, tensor in ipairs(tensorArray) do
		
		if checkIfIsAutomaticDifferentiationTensor(tensor) then

			return tensor.new(result, PartialDerivativeFunction, tensorArray)
			
		end
		
	end

end

function AHAAutomaticDifferentiatonTensor:subtract(...)

	local tensorArray = {self, ...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	for i, tensor in ipairs(tensorArray) do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	end

	local result = AqwamTensorLibrary:subtract(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local collapsedDerivativeTensor = collapseTensor(derivativeTensor, dimensionSizeArrayArray[i])

				tensor:differentiate(collapsedDerivativeTensor) 

			end

		end

	end

	for _, tensor in ipairs(tensorArray) do

		if checkIfIsAutomaticDifferentiationTensor(tensor) then

			return tensor.new(result, PartialDerivativeFunction, tensorArray)

		end

	end

end

function AHAAutomaticDifferentiatonTensor:multiply(...)

	local tensorArray = {self, ...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	for i, tensor in ipairs(tensorArray) do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	end

	local result = AqwamTensorLibrary:multiply(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do
			
			local remainingTensorArray = {}
			
			for j, tensor in ipairs(tensorArray) do

				if (j ~= i) then table.insert(remainingTensorArray, tensor) end
					
			end
			
			local currentDerivativeTensor = AqwamTensorLibrary:multiply(derivativeTensor, table.unpack(remainingTensorArray))

			local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

			tensor:differentiate(collapsedCurrentDerivativeTensor)

		end

	end

	for _, tensor in ipairs(tensorArray) do

		if checkIfIsAutomaticDifferentiationTensor(tensor) then

			return tensor.new(result, PartialDerivativeFunction, tensorArray)

		end

	end

end

function AHAAutomaticDifferentiatonTensor:divide(...)

	local tensorArray = {self, ...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	for i, tensor in ipairs(tensorArray) do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	end

	local result = AqwamTensorLibrary:divide(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			local remainingTensorArray = {}

			for j, tensor in ipairs(tensorArray) do

				if (j ~= i) then table.insert(remainingTensorArray, tensor) end

			end

			local currentDerivativeTensor = AqwamTensorLibrary:multiply(derivativeTensor, table.unpack(remainingTensorArray))

			local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArrayArray[i])

			tensor:differentiate(collapsedCurrentDerivativeTensor)

		end

	end

	for _, tensor in ipairs(tensorArray) do

		if checkIfIsAutomaticDifferentiationTensor(tensor) then

			return tensor.new(result, PartialDerivativeFunction, tensorArray)

		end

	end

end

function AHAAutomaticDifferentiatonTensor:sum(dimension)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:sum(self, dimension)

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

	return self.new(result, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:unaryMinus()

	local result = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return self.new(result, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:power(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:power(self, other)

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

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:logarithm(other)

	local selfDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local otherDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(other)

	local result = AqwamTensorLibrary:logarithm(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, selfDimensionSizeArray)

			local partialDerivativeTensor

			if (other) then

				local partialDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, self, other)

			else

				local partialDerivativeFunctionToApply = function (number) return (1 / number) end

				partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, self)

			end

			self:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

		if checkIfIsAutomaticDifferentiationTensor(other) then

			local collapsedDerivativeTensor = collapseTensor(derivativeTensor, otherDimensionSizeArray)

			local partialDerivativeFunctionToApply = function (number, base) return -(math.log(number) / (base * math.pow(math.log(base), 2))) end

			local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, self, other)

			other:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, collapsedDerivativeTensor)) 

		end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:dotProduct(other) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local result = AqwamTensorLibrary:dotProduct(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		local otherNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(other)
		local selfNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(self)

		local transposedOther = AqwamTensorLibrary:transpose(other, {otherNumberOfDimensions - 1, otherNumberOfDimensions})
		local transposedSelf = AqwamTensorLibrary:transpose(self, {selfNumberOfDimensions - 1, selfNumberOfDimensions})

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:dotProduct(derivativeTensor, transposedOther)) end
		if checkIfIsAutomaticDifferentiationTensor(other) then other:differentiate(AqwamTensorLibrary:dotProduct(transposedSelf, derivativeTensor)) end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(result, PartialDerivativeFunction, {self, other})

	else

		return other.new(result, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiatonTensor:extract(originDimensionIndexArray, targetDimensionIndexArray)
	
	local originalTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)
	
	local result = AqwamTensorLibrary:extract(self, originDimensionIndexArray, targetDimensionIndexArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local originDimensionIndexArray = originDimensionIndexArray

		local targetDimensionIndexArray = targetDimensionIndexArray

		local numberOfDimensions = #originalTensorDimensionSizeArray

		local headPaddingDimensionSizeArray = {}

		local tailPaddingDimensionSizeArray = {}

		for dimension = 1, numberOfDimensions, 1 do

			headPaddingDimensionSizeArray[dimension] = originDimensionIndexArray[dimension] - 1

			tailPaddingDimensionSizeArray[dimension] = originalTensorDimensionSizeArray[dimension] - targetDimensionIndexArray[dimension]

		end

		for dimension = numberOfDimensions, 1, -1 do

			local derivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

			local headPaddingDimensionSize = headPaddingDimensionSizeArray[dimension]

			local tailPaddingDimensionSize = tailPaddingDimensionSizeArray[dimension]

			if (headPaddingDimensionSize >= 1) then

				local tensorHeadPaddingDimensionSizeArray = table.clone(derivativeTensorDimensionSizeArray)

				tensorHeadPaddingDimensionSizeArray[dimension] = headPaddingDimensionSize

				local headPaddingTensor = AqwamTensorLibrary:createTensor(tensorHeadPaddingDimensionSizeArray)

				derivativeTensor = AqwamTensorLibrary:concatenate(headPaddingTensor, derivativeTensor, dimension)

			end

			if (tailPaddingDimensionSize >= 1) then

				local tensorTailPaddingDimensionSizeArray = table.clone(derivativeTensorDimensionSizeArray)

				tensorTailPaddingDimensionSizeArray[dimension] = tailPaddingDimensionSize

				local tailPaddingTensor = AqwamTensorLibrary:createTensor(tensorTailPaddingDimensionSizeArray)

				derivativeTensor = AqwamTensorLibrary:concatenate(derivativeTensor, tailPaddingTensor, dimension)

			end

		end
		
		self:differentiate(derivativeTensor)

	end

	return self.new(result, PartialDerivativeFunction, {self})
	
end

function AHAAutomaticDifferentiatonTensor.concatenate(...)
	
	local tensorArray = {...}
	
	local numberOfArguments = #tensorArray
	
	local dimensionIndex = tensorArray[numberOfArguments]
	
	if (type(dimensionIndex) ~= "number") then error("The final argument must be a number in order for it to be used as dimension index.") end
	
	table.remove(tensorArray, numberOfArguments)
	
	local result

	for i, tensor in ipairs(tensorArray) do

		if (i > 1) then

			result = AqwamTensorLibrary:concatenate(result, tensor, dimensionIndex)

		else

			result = tensor

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)
		
		local extractedDerivativeTensorArray = {}

		local derivativeTensorDimensionArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local originDimensionIndexArray = table.create(#derivativeTensorDimensionArray, 1)

		local targetDimensionIndexArray = table.clone(derivativeTensorDimensionArray)

		targetDimensionIndexArray[dimensionIndex] = 0

		for _, tensor in ipairs(tensorArray) do

			local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

			targetDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex] - 1

			local extractedDerivativeTensor = AqwamTensorLibrary:extract(derivativeTensor, originDimensionIndexArray, targetDimensionIndexArray)

			originDimensionIndexArray[dimensionIndex] = originDimensionIndexArray[dimensionIndex] + dimensionSizeArray[dimensionIndex]

			table.insert(extractedDerivativeTensorArray, extractedDerivativeTensor)

		end

		for i, tensor in ipairs(tensorArray) do
			
			if checkIfIsAutomaticDifferentiationTensor(tensor) then tensor:differentiate(extractedDerivativeTensorArray[i]) end
			
		end

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, tensorArray)

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiatonTensor:transpose(dimensionIndexArray)

	local result = AqwamTensorLibrary:transpose(self, dimensionIndexArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:transpose(derivativeTensor, dimensionIndexArray)) end

	end

	return self.new(result, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:flatten(dimensionArray)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:flatten(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, dimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:reshape(dimensionSizeArray)

	local originalDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local result = AqwamTensorLibrary:reshape(self, dimensionSizeArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, originalDimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiatonTensor:permute(dimensionArray)
	
	local originalDimensionArray = createOriginalDimensionArray(dimensionArray)

	local result = AqwamTensorLibrary:permute(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:permute(derivativeTensor, originalDimensionArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiatonTensor.new(result, PartialDerivativeFunction, {self})

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
	
	local tensorArray = self.tensorArray

	if (areDescendantsDestroyed) and (tensorArray) then
		
		for _, tensor in ipairs(tensorArray) do
			
			if checkIfIsAutomaticDifferentiationTensor(tensor) then tensor:destroy(true) end
			
		end

	end

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AHAAutomaticDifferentiatonTensor