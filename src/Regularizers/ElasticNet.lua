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

local BaseRegularizer = require(script.Parent.BaseRegularizer)

local ElasticNet = {}

ElasticNet.__index = ElasticNet

setmetatable(ElasticNet, BaseRegularizer)

function ElasticNet.new(parameterDictionary)
	
	local NewElasticNet = BaseRegularizer.new(parameterDictionary)
	
	setmetatable(NewElasticNet, ElasticNet)
	
	NewElasticNet:setName("ElasticNet")
	
	NewElasticNet:setCalculateFunction(function(weightTensor)
		
		local signTensor = AqwamTensorLibrary:applyFunction(math.sign, weightTensor)

		local regularizationTensorPart1 = AqwamTensorLibrary:multiply(NewElasticNet.lambda, signTensor)

		local regularizationTensorPart2 = AqwamTensorLibrary:multiply(2, NewElasticNet.lambda, weightTensor)

		return AqwamTensorLibrary:add(regularizationTensorPart1, regularizationTensorPart2)
		
	end)
	
	return NewElasticNet
	
end

return ElasticNet
