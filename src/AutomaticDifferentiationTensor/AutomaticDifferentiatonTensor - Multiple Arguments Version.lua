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

local AHAAutomaticDifferentiationTensor = {}

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

function AHAAutomaticDifferentiationTensor.new(tensor, PartialDerivativeFunction, inputTensorArray)

	local self = setmetatable({}, AHAAutomaticDifferentiationTensor)

	self.tensor = tensor

	self.PartialDerivativeFunction = PartialDerivativeFunction

	self.inputTensorArray = inputTensorArray

	self.totalDerivativeTensor = nil

	return self

end

function AHAAutomaticDifferentiationTensor.radian(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.rad, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local radiansPerDegree = math.pi / 180

		tensor:differentiate(AqwamTensorLibrary:multiply(radiansPerDegree, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiationTensor.degree(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.deg, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local degreesPerRadian = 180 / math.pi

		tensor:differentiate(AqwamTensorLibrary:multiply(degreesPerRadian, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiationTensor.sin(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.sin, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiationTensor.cos(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.cos, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return -math.sin(radian) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiationTensor.tan(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.tan, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local partialDerivativeFunctionToApply = function (radian) return math.pow((1 / math.cos(radian)), 2) end

		local partialDerivativeTensor = AqwamTensorLibrary:applyFunction(partialDerivativeFunctionToApply, tensor)

		tensor:differentiate(AqwamTensorLibrary:multiply(partialDerivativeTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiationTensor.exponent(tensor)

	local resultTensor = AqwamTensorLibrary:applyFunction(math.exp, tensor)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		tensor:differentiate(AqwamTensorLibrary:multiply(resultTensor, derivativeTensor))

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function AHAAutomaticDifferentiationTensor.logarithm(numberTensor, baseTensor)

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

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {numberTensor, baseTensor})

end

function AHAAutomaticDifferentiationTensor.clamp(tensor, lowerBoundTensor, upperBoundTensor)

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

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor, lowerBoundTensor, upperBoundTensor})

end

function AHAAutomaticDifferentiationTensor.maximum(...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 2, numberOfTensors, 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i])

		expandedTensorArray[i - 1], expandedTensorArray[i] = AqwamTensorLibrary:broadcast(tensorArray[i - 1], tensorArray[i])

	end

	local resultTensor = AqwamTensorLibrary:applyfunction(math.max, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local functionToApply = function(derivativeValue, ...)
					
					local valueArray = {...}

					local isMaximum = false

					local highestValue = -math.huge

					for j, value in ipairs(valueArray) do

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

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

function AHAAutomaticDifferentiationTensor.minimum(...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local dimensionSizeArrayArray = {}

	local expandedTensorArray = {}

	dimensionSizeArrayArray[1] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[1])

	for i = 2, numberOfTensors, 1 do

		dimensionSizeArrayArray[i] = AqwamTensorLibrary:getDimensionSizeArray(tensorArray[i])

		expandedTensorArray[i - 1], expandedTensorArray[i] = AqwamTensorLibrary:broadcast(tensorArray[i - 1], tensorArray[i])

	end

	local resultTensor = AqwamTensorLibrary:applyfunction(math.min, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local functionToApply = function(derivativeValue, ...)
					
					local valueArray = {...}

					local isMinimum = false

					local lowestValue = -math.huge

					for j, value in ipairs(valueArray) do

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

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__eq(other)

	return AqwamTensorLibrary:isSameTensor(self, other)

end

function AHAAutomaticDifferentiationTensor:__add(other)

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

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:__sub(other)

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

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:__mul(other)

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

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:__div(other)

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

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:__unm()

	local resultTensor = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return self.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:__pow(other)

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

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:add(...)

	local tensorArray = {self, ...}

	local resultTensor = AqwamTensorLibrary:add(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedDerivativeTensor = collapseTensor(derivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

function AHAAutomaticDifferentiationTensor:subtract(...)

	local tensorArray = {self, ...}

	local resultTensor = AqwamTensorLibrary:subtract(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedDerivativeTensor = collapseTensor(derivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedDerivativeTensor) 

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

function AHAAutomaticDifferentiationTensor:multiply(...)

	local tensorArray = {self, ...}

	local resultTensor = AqwamTensorLibrary:multiply(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local remainingTensorArray = {}

				for j, tensor in ipairs(tensorArray) do

					if (j ~= i) then table.insert(remainingTensorArray, tensor) end

				end

				local currentDerivativeTensor = AqwamTensorLibrary:multiply(derivativeTensor, table.unpack(remainingTensorArray))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedCurrentDerivativeTensor)

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

function AHAAutomaticDifferentiationTensor:divide(...)

	local tensorArray = {self, ...}

	local resultTensor = AqwamTensorLibrary:divide(self, ...)

	local PartialDerivativeFunction = function(derivativeTensor)

		for i, tensor in ipairs(tensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then

				local remainingTensorArray = {}

				for j, tensor in ipairs(tensorArray) do

					if (j ~= i) then table.insert(remainingTensorArray, tensor) end

				end

				local currentDerivativeTensor = AqwamTensorLibrary:multiply(derivativeTensor, table.unpack(remainingTensorArray))

				local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

				local collapsedCurrentDerivativeTensor = collapseTensor(currentDerivativeTensor, dimensionSizeArray)

				tensor:differentiate(collapsedCurrentDerivativeTensor)

			end

		end

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

function AHAAutomaticDifferentiationTensor:sum(dimension)

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

function AHAAutomaticDifferentiationTensor:unaryMinus()

	local resultTensor = AqwamTensorLibrary:unaryMinus(self)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:unaryMinus(derivativeTensor)) end

	end

	return self.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:power(other)

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

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:dotProduct(other) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local resultTensor = AqwamTensorLibrary:dotProduct(self, other)

	local PartialDerivativeFunction = function(derivativeTensor)

		local otherNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(other)
		local selfNumberOfDimensions = #AqwamTensorLibrary:getDimensionSizeArray(self)

		local transposedOther = AqwamTensorLibrary:transpose(other, {otherNumberOfDimensions - 1, otherNumberOfDimensions})
		local transposedSelf = AqwamTensorLibrary:transpose(self, {selfNumberOfDimensions - 1, selfNumberOfDimensions})

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:dotProduct(derivativeTensor, transposedOther)) end
		if checkIfIsAutomaticDifferentiationTensor(other) then other:differentiate(AqwamTensorLibrary:dotProduct(transposedSelf, derivativeTensor)) end

	end

	if checkIfIsAutomaticDifferentiationTensor(self) then

		return self.new(resultTensor, PartialDerivativeFunction, {self, other})

	else

		return other.new(resultTensor, PartialDerivativeFunction, {self, other})

	end

end

function AHAAutomaticDifferentiationTensor:extract(originDimensionIndexArray, targetDimensionIndexArray)

	local originalTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local resultTensor = AqwamTensorLibrary:extract(self, originDimensionIndexArray, targetDimensionIndexArray)

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

	return self.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor.concatenate(...)

	local tensorArray = {...}

	local numberOfArguments = #tensorArray

	local dimensionIndex = tensorArray[numberOfArguments]

	if (type(dimensionIndex) ~= "number") then error("The final argument must be a number in order for it to be used as dimension index.") end

	table.remove(tensorArray, numberOfArguments)

	local resultTensor

	for i, tensor in ipairs(tensorArray) do

		if (i > 1) then

			resultTensor = AqwamTensorLibrary:concatenate(resultTensor, tensor, dimensionIndex)

		else

			resultTensor = tensor

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

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, tensorArray)

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:transpose(dimensionIndexArray)

	local resultTensor = AqwamTensorLibrary:transpose(self, dimensionIndexArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if checkIfIsAutomaticDifferentiationTensor(self) then self:differentiate(AqwamTensorLibrary:transpose(derivativeTensor, dimensionIndexArray)) end

	end

	return self.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:flatten(dimensionArray)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local resultTensor = AqwamTensorLibrary:flatten(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, dimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:reshape(dimensionSizeArray)

	local originalDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

	local resultTensor = AqwamTensorLibrary:reshape(self, dimensionSizeArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:reshape(derivativeTensor, originalDimensionSizeArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:permute(dimensionArray)

	local originalDimensionArray = createOriginalDimensionArray(dimensionArray)

	local resultTensor = AqwamTensorLibrary:permute(self, dimensionArray)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:permute(derivativeTensor, originalDimensionArray)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:flip(dimension)

	local resultTensor = AqwamTensorLibrary:flip(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		derivativeTensor = AqwamTensorLibrary:flip(derivativeTensor, dimension)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:mean(dimension)

	local resultTensor = AqwamTensorLibrary:mean(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(self)[dimension]

		derivativeTensor = AqwamTensorLibrary:divide(derivativeTensor, dimensionSize)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})	

end

function AHAAutomaticDifferentiationTensor:standardDeviation(dimension)

	local resultTensor = AqwamTensorLibrary:standardDeviation(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local dimensionSize = AqwamTensorLibrary:getDimensionSizeArray(self)[dimension]

		local chainRuleFirstDerivativeTensorPart1 = AqwamTensorLibrary:multiply(2, resultTensor, dimensionSize)

		derivativeTensor = AqwamTensorLibrary:divide(derivativeTensor, chainRuleFirstDerivativeTensorPart1)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})	

end

function AHAAutomaticDifferentiationTensor:zScoreNormalization(dimension)

	local resultTensor = AqwamTensorLibrary:standardDeviation(self, dimension)

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local standardDeviationTensor = AqwamTensorLibrary:standardDeviation(self, dimension)

		derivativeTensor = AqwamTensorLibrary:divide(derivativeTensor, standardDeviationTensor)

		self:differentiate(derivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {self})	

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:expandDimensionSizes(targetDimensionSizeArray)

	local resultTensor = AqwamTensorLibrary:expandDimensionSizes(self, targetDimensionSizeArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self)

		local chainRuleFirstDerivativeTensor = firstDerivativeTensor

		for dimension, dimensionSize in ipairs(tensorDimensionSizeArray) do

			if (dimensionSize == 1) and (targetDimensionSizeArray[dimension] > 1) then

				chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, dimension)

			end

		end

		self:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialFirstDerivativeFunction, {self})

end

function AHAAutomaticDifferentiationTensor:expandNumberOfDimensions(dimensionSizeToAddArray)

	local resultTensor = AqwamTensorLibrary:expandNumberOfDimensions(self, dimensionSizeToAddArray)

	local PartialFirstDerivativeFunction = function(firstDerivativeTensor)

		if (not AHAAutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(self)) then return end

		local numberOfDimensionsToSum = #dimensionSizeToAddArray

		local chainRuleFirstDerivativeTensor = firstDerivativeTensor

		for i = 1, numberOfDimensionsToSum, 1 do chainRuleFirstDerivativeTensor = AqwamTensorLibrary:sum(chainRuleFirstDerivativeTensor, 1)[1] end -- Remove the first dimension as it is redundant and does not carry any values. If it is not removed, this tensor might broadcast its dimension size elsewhere like during the gradient descent.

		self:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AHAAutomaticDifferentiationTensor.new(resultTensor, PartialFirstDerivativeFunction, {self})

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:isAutomaticDifferentiationTensor()

	return true

end

function AHAAutomaticDifferentiationTensor:differentiate(derivativeTensor)
	
	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(self.tensor)
	
	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	if (not derivativeTensor) then
		
		if (tensorNumberOfDimensions >= 1) then
			
			derivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray, 1)
			
		else
			
			derivativeTensor = 1
			
		end
		
	else
		
		local derivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local derivativeTensorNumberOfDimensions = #derivativeTensorDimensionSizeArray
		
		if (derivativeTensorNumberOfDimensions ~= 0) then

			if (derivativeTensorNumberOfDimensions ~= tensorNumberOfDimensions) then error("Unable to differentiate. The derivative tensor has " .. derivativeTensorNumberOfDimensions .. ", but the original tensor has " .. tensorNumberOfDimensions .. ".") end

			for dimension, derivativeTensorDimensionSize in ipairs(derivativeTensorDimensionSizeArray) do

				local tensorDimensionSize = tensorDimensionSizeArray[dimension]

				if (derivativeTensorDimensionSize ~= tensorDimensionSize) then

					error("Unable to differentiate. The derivative tensor has a dimension size of " .. derivativeTensorDimensionSize .. " at dimension " .. dimension .. ", but the original tensor has " .. tensorDimensionSize .. ".")

				end

			end

		end

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

function AHAAutomaticDifferentiationTensor:copy()

	return deepCopyTable(self)

end

function AHAAutomaticDifferentiationTensor:getTensor(doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		return self.tensor

	else

		return deepCopyTable(self.tensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTensor(tensor, doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		self.tensor = tensor

	else

		self.tensor = deepCopyTable(tensor)

	end

end

function AHAAutomaticDifferentiationTensor:getTotalDerivativeTensor(doNotDeepCopyTable)

	if (doNotDeepCopyTable) then 

		return self.totalDerivativeTensor

	else

		return deepCopyTable(self.totalDerivativeTensor)

	end

end

function AHAAutomaticDifferentiationTensor:setTotalDerivativeTensor(totalDerivativeTensor, doNotDeepCopyTable)

	if (doNotDeepCopyTable) then

		self.totalDerivativeTensor = totalDerivativeTensor

	else

		self.totalDerivativeTensor = deepCopyTable(totalDerivativeTensor)

	end

end

--------------------------------------------------------------------------------------

function AHAAutomaticDifferentiationTensor:__tostring()
	
	local tensor = self.tensor

	if (type(tensor) == "table") then

		return AqwamTensorLibrary:generateTensorString(tensor)

	else

		return tostring(tensor)	

	end

end

function AHAAutomaticDifferentiationTensor:__len()

	local tensor = self.tensor

	if (type(tensor) == "table") then

		return #tensor 

	else

		return 0

	end

end

function AHAAutomaticDifferentiationTensor:__index(index)

	if (type(index) == "number") then

		local tensor = self.tensor

		if (type(tensor) == "table") then

			return rawget(tensor, index)

		else

			return tensor

		end

	else

		return rawget(AHAAutomaticDifferentiationTensor, index)

	end

end

function AHAAutomaticDifferentiationTensor:__newindex(index, value)

	rawset(self, index, value)

end

function AHAAutomaticDifferentiationTensor:destroy(areDescendantsDestroyed)

	local inputTensorArray = self.inputTensorArray

	if (areDescendantsDestroyed) and (inputTensorArray) then

		for _, tensor in ipairs(inputTensorArray) do

			if checkIfIsAutomaticDifferentiationTensor(tensor) then tensor:destroy(true) end

		end

	end

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AHAAutomaticDifferentiationTensor