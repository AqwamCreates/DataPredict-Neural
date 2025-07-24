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

local BaseModel = require(script.Parent.BaseModel)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local GenerativeAdversarialNetworkBaseModel = {}

GenerativeAdversarialNetworkBaseModel.__index = GenerativeAdversarialNetworkBaseModel

setmetatable(GenerativeAdversarialNetworkBaseModel, BaseModel)

local discriminatorLossFunction = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel)) end

local generatorLossFunction = function (discriminatorGeneratedLabel) return math.log(1 - discriminatorGeneratedLabel) end

function GenerativeAdversarialNetworkBaseModel.new()
	
	local NewGenerativeAdversarialNetworkBaseModel = BaseModel.new()
	
	setmetatable(NewGenerativeAdversarialNetworkBaseModel, GenerativeAdversarialNetworkBaseModel)
	
	NewGenerativeAdversarialNetworkBaseModel:setName("GenerativeAdversarialNetworkBaseModel")
	
	return NewGenerativeAdversarialNetworkBaseModel
	
end

function GenerativeAdversarialNetworkBaseModel:setDiscriminatorLossFunction(DiscriminatorLossFunction)
	
	self.DiscriminatorLossFunction = DiscriminatorLossFunction
	
end

function GenerativeAdversarialNetworkBaseModel:setGeneratorLossFunction(GeneratorLossFunction)
	
	self.GeneratorLossFunction = GeneratorLossFunction
	
end

function GenerativeAdversarialNetworkBaseModel:calculateDiscriminatorLossTensor(evaluatedRealTensor, evaluatedGeneratedTensor)

	return self.DiscriminatorLossFunction(evaluatedRealTensor, evaluatedGeneratedTensor)

end

function GenerativeAdversarialNetworkBaseModel:calculateGeneratorLossTensor(evaluatedGeneratedTensor)
	
	return self.GeneratorLossFunction(evaluatedGeneratedTensor)
	
end

function GenerativeAdversarialNetworkBaseModel:updateDiscriminator(discriminatorLossTensor, numberOfData)

	local DiscriminatorModel = self.DiscriminatorModel

	local discriminatorSeedFeatureTensor = self.discriminatorSeedFeatureTensor

	if (not DiscriminatorModel) then error("No discriminator model.") end

	if (not discriminatorSeedFeatureTensor) then error("No discriminator seed feature tensor.") end

	DiscriminatorModel:forwardPropagate(discriminatorSeedFeatureTensor)

	DiscriminatorModel:update(discriminatorLossTensor, numberOfData)

end

function GenerativeAdversarialNetworkBaseModel:updateGenerator(generatorLossTensor, numberOfData)

	local GeneratorModel = self.GeneratorModel

	local generatorSeedFeatureTensor = self.generatorSeedFeatureTensor

	if (not GeneratorModel) then error("No generator model.") end

	if (not generatorSeedFeatureTensor) then error("No generator seed feature tensor.") end

	GeneratorModel:forwardPropagate(generatorSeedFeatureTensor, true)

	GeneratorModel:update(generatorLossTensor, true)

end

function GenerativeAdversarialNetworkBaseModel:evaluate(featureTensor)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	if (not DiscriminatorModel) then error("No discriminator model.") end

	return DiscriminatorModel:forwardPropagate(featureTensor)

end

function GenerativeAdversarialNetworkBaseModel:generate(noiseFeatureTensor)
	
	local GeneratorModel = self.GeneratorModel
	
	if (not GeneratorModel) then error("No generator model.") end

	return GeneratorModel:forwardPropagate(noiseFeatureTensor)

end

function GenerativeAdversarialNetworkBaseModel:setDiscriminatorModel(DiscriminatorModel)

	self.DiscriminatorModel = DiscriminatorModel

end

function GenerativeAdversarialNetworkBaseModel:setGeneratorModel(GeneratorModel)

	self.GeneratorModel = GeneratorModel

end

function GenerativeAdversarialNetworkBaseModel:getDiscriminatorModel()

	return self.DiscriminatorModel

end

function GenerativeAdversarialNetworkBaseModel:getGeneratorModel()

	return self.GeneratorModel

end

function GenerativeAdversarialNetworkBaseModel:setDiscriminatorSeedFeatureTensor(discriminatorSeedFeatureTensor, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.discriminatorSeedFeatureTensor = discriminatorSeedFeatureTensor
		
	else
		
		self.discriminatorSeedFeatureTensor = self:deepCopyTable(discriminatorSeedFeatureTensor)
		
	end

end

function GenerativeAdversarialNetworkBaseModel:setGeneratorSeedFeatureTensor(generatorSeedFeatureTensor, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.generatorSeedFeatureTensor = generatorSeedFeatureTensor

	else

		self.generatorSeedFeatureTensor = self:deepCopyTable(generatorSeedFeatureTensor)

	end

end

function GenerativeAdversarialNetworkBaseModel:getDiscriminatorSeedFeatureTensor(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.discriminatorSeedFeatureTensor
			
	else
		
		return self:deepCopyTable(self.discriminatorSeedFeatureTensor)
		
	end

end

function GenerativeAdversarialNetworkBaseModel:getGeneratorSeedFeatureTensor(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.generatorSeedFeatureTensor

	else

		return self:deepCopyTable(self.generatorSeedFeatureTensor)

	end

end

return GenerativeAdversarialNetworkBaseModel