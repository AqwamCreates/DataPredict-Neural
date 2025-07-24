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

local GenerativeAdversarialNetworkBaseModel = require(script.Parent.GenerativeAdversarialNetworkBaseModel)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local GenerativeAdversarialNetwork = {}

GenerativeAdversarialNetwork.__index = GenerativeAdversarialNetwork

setmetatable(GenerativeAdversarialNetwork, GenerativeAdversarialNetworkBaseModel)

function GenerativeAdversarialNetwork.new()
	
	local NewGenerativeAdversarialNetwork = GenerativeAdversarialNetworkBaseModel.new()
	
	setmetatable(NewGenerativeAdversarialNetwork, GenerativeAdversarialNetwork)
	
	NewGenerativeAdversarialNetwork:setName("GenerativeAdversarialNetwork")
	
	NewGenerativeAdversarialNetwork:setDiscriminatorLossFunction(function(evaluatedRealTensor, evaluatedGeneratedTensor)

		local discriminatorLossFunction = function (evaluatedRealValue, evaluatedGeneratedValue) return -(math.log(evaluatedRealValue) + math.log(1 - evaluatedGeneratedValue)) end

		local discriminatorLossTensor = AqwamTensorLibrary:applyFunction(discriminatorLossFunction, evaluatedRealTensor, evaluatedGeneratedTensor)

		return discriminatorLossTensor

	end)
	
	NewGenerativeAdversarialNetwork:setGeneratorLossFunction(function(evaluatedGeneratedTensor)
		
		local generatorLossFunction = function (evaluatedGeneratedValue) return math.log(1 - evaluatedGeneratedValue) end
		
		local generatorLossTensor = AqwamTensorLibrary:applyFunction(generatorLossFunction, evaluatedGeneratedTensor)
		
		return generatorLossTensor
		
	end)
	
	return NewGenerativeAdversarialNetwork
	
end

return GenerativeAdversarialNetwork
