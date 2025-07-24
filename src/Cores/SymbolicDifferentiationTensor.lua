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

local AHASymbolicDifferentiationTensor = {}

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

--------------------------------------------------------------------------------------

function AHASymbolicDifferentiationTensor.add(...)
	
	local self = setmetatable({}, AHASymbolicDifferentiationTensor)
	
	local tensor = AqwamTensorLibrary:add(...)
	
	self.tensor = tensor
	
	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getSize(tensor), 1) end
	
	return self
	
end

function AHASymbolicDifferentiationTensor.subtract(...)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)

	local tensor = AqwamTensorLibrary:subtract(...)

	self.tensor = tensor

	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getSize(tensor), 1) end

	return self

end

function AHASymbolicDifferentiationTensor.multiply(...)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)
	
	local tensor = AqwamTensorLibrary:multiply(...)

	self.tensor = tensor

	self.FirstDerivativeFunction = function (tensor) return AqwamTensorLibrary:divide(tensor, tensor) end

	return self

end

function AHASymbolicDifferentiationTensor.divide(...)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)

	local tensor = AqwamTensorLibrary:divide(...)

	self.tensor = tensor

	self.FirstDerivativeFunction = function (tensor) return AqwamTensorLibrary:divide(tensor, tensor) end

	return self

end

function AHASymbolicDifferentiationTensor.power(baseTensor, exponentTensor)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)
	
	local firstDerivativeFunctionToApply = function (base, exponent)
		
		return math.pow((exponent * base), (exponent - 1))
		
	end

	self.tensor = AqwamTensorLibrary:applyFunction(math.pow, baseTensor, exponentTensor)

	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:applyFunction(firstDerivativeFunctionToApply, baseTensor, exponentTensor) end

	return self

end

function AHASymbolicDifferentiationTensor.exponent(exponentTensor)
	
	local self = setmetatable({}, AHASymbolicDifferentiationTensor)
	
	local tensor = AqwamTensorLibrary:applyFunction(math.exp, exponentTensor)

	self.tensor = tensor

	self.FirstDerivativeFunction = function () return tensor end

	return self
	
end

function AHASymbolicDifferentiationTensor.log(numberTensor, baseTensor)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)

	local firstDerivativeFunctionToApply
	
	if (baseTensor) then
		
		firstDerivativeFunctionToApply = function (number, base) return (1 / (number * math.log(base))) end
		
	else
		
		firstDerivativeFunctionToApply = function (number) return (1 / number) end
		
	end

	self.tensor = AqwamTensorLibrary:applyFunction(math.log, numberTensor, baseTensor)

	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:applyFunction(firstDerivativeFunctionToApply, numberTensor, baseTensor) end

	return self

end

function AHASymbolicDifferentiationTensor.sin(radianTensor)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)

	self.tensor = AqwamTensorLibrary:applyFunction(math.sin, radianTensor)

	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:applyFunction(math.cos, radianTensor) end

	return self

end

function AHASymbolicDifferentiationTensor.cos(radianTensor)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)
	
	local firstDerivativeFunctionToApply = function (radian) return -math.sin(radian) end
	
	self.tensor = AqwamTensorLibrary:applyFunction(math.cos, radianTensor)

	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:applyFunction(firstDerivativeFunctionToApply, radianTensor) end

	return self

end

function AHASymbolicDifferentiationTensor.tan(radianTensor)

	local self = setmetatable({}, AHASymbolicDifferentiationTensor)

	local firstDerivativeFunctionToApply = function (radian) return math.pow((1 / math.cos(radian)), 2) end

	self.tensor = AqwamTensorLibrary:applyFunction(math.tan, radianTensor)

	self.FirstDerivativeFunction = function () return AqwamTensorLibrary:applyFunction(firstDerivativeFunctionToApply, radianTensor) end

	return self

end

function AHASymbolicDifferentiationTensor.dotProduct(tensor1, tensor2)
	
	local self = setmetatable({}, AHASymbolicDifferentiationTensor)

	self.tensor = AqwamTensorLibrary:dotProduct(tensor1, tensor2)

	self.FirstDerivativeFunction = function (firstDerivativeTensor1, firstDerivativeTensor2)
		
		local firstPartFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(tensor1, firstDerivativeTensor2)	
		
		local secondPartFirstDerivativeTensor = AqwamTensorLibrary:dotProduct(firstDerivativeTensor1, tensor2)
		
		return AqwamTensorLibrary:add(firstPartFirstDerivativeTensor, secondPartFirstDerivativeTensor)
		
	end

	return self
	
end

--------------------------------------------------------------------------------------

function AHASymbolicDifferentiationTensor:getTensor(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.tensor

	else

		return deepCopyTable(self.tensor)

	end

end

function AHASymbolicDifferentiationTensor:setTensor(tensor, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.tensor = tensor

	else

		self.tensor = deepCopyTable(tensor)

	end

end

function AHASymbolicDifferentiationTensor:getFirstDerivativeTensor(...)
	
	return self.FirstDerivativeFunction(...)
	
end

--------------------------------------------------------------------------------------

function AHASymbolicDifferentiationTensor:__tostring()
	
	return AqwamTensorLibrary:generateTensorString(self.tensor)
	
end

function AHASymbolicDifferentiationTensor:__len()

	return #self.tensor

end

function AHASymbolicDifferentiationTensor:__index(index)

	if (type(index) == "number") then

		return rawget(self.tensor, index)

	else

		return rawget(AHASymbolicDifferentiationTensor, index)

	end

end

function AHASymbolicDifferentiationTensor:__newindex(index, value)

	rawset(self, index, value)

end

function AHASymbolicDifferentiationTensor:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

--------------------------------------------------------------------------------------

return AHASymbolicDifferentiationTensor