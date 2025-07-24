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

local WassersteinGenerativeAdversarialNetwork = {}

WassersteinGenerativeAdversarialNetwork.__index = WassersteinGenerativeAdversarialNetwork

setmetatable(WassersteinGenerativeAdversarialNetwork, GenerativeAdversarialNetworkBaseModel)

function WassersteinGenerativeAdversarialNetwork.new()
	
	local NewWassersteinGenerativeAdversarialNetwork = GenerativeAdversarialNetworkBaseModel.new()
	
	setmetatable(NewWassersteinGenerativeAdversarialNetwork, WassersteinGenerativeAdversarialNetwork)
	
	NewWassersteinGenerativeAdversarialNetwork:setName("WassersteinGenerativeAdversarialNetwork")
	
	NewWassersteinGenerativeAdversarialNetwork:setDiscriminatorLossFunction(function(evaluatedRealTensor, evaluatedGeneratedTensor)

		local discriminatorLossFunction = function (evaluatedRealValue, evaluatedGeneratedValue) return -(evaluatedRealValue - evaluatedGeneratedValue) end

		local discriminatorLossTensor = AqwamTensorLibrary:applyFunction(discriminatorLossFunction, evaluatedRealTensor, evaluatedGeneratedTensor)

		return discriminatorLossTensor

	end)
	
	NewWassersteinGenerativeAdversarialNetwork:setGeneratorLossFunction(function(evaluatedGeneratedTensor)
		
		local generatorLossTensor = AqwamTensorLibrary:unaryMinus(evaluatedGeneratedTensor)
		
		return generatorLossTensor
		
	end)
	
	return NewWassersteinGenerativeAdversarialNetwork
	
end

return WassersteinGenerativeAdversarialNetwork
