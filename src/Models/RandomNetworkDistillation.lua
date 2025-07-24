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

local BaseModel = require(script.Parent.BaseModel)

local RandomNetworkDistillation = {}

RandomNetworkDistillation.__index = RandomNetworkDistillation

function RandomNetworkDistillation.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewRandomNetworkDistillation = BaseModel.new(parameterDictionary)
	
	setmetatable(NewRandomNetworkDistillation, RandomNetworkDistillation)
	
	NewRandomNetworkDistillation:setName("RandomNetworkDistillation")
	
	NewRandomNetworkDistillation.Model = parameterDictionary.Model
	
	NewRandomNetworkDistillation.TargetWeightTensorArray = parameterDictionary.TargetWeightTensorArray
	
	NewRandomNetworkDistillation.PredictorWeightTensorArray = parameterDictionary.PredictorWeightTensorArray
	
	return NewRandomNetworkDistillation
	
end

function RandomNetworkDistillation:setModel(Model)
	
	self.Model = Model
	
end

function RandomNetworkDistillation:getModel(Model)
	
	return self.Model
	
end

function RandomNetworkDistillation:generate(featureVector)
	
	local Model = self.Model
	
	if (not Model) then error("No model!") end
	
	local TargetWeightTensorArray = self.TargetWeightTensorArray
	
	local PredictorWeightTensorArray = self.PredictorWeightTensorArray
	
	Model:setWeightTensorArray(TargetWeightTensorArray, true)
	
	local targetTensor = Model:predict(featureVector, true)
	
	Model:setWeightTensorArray(PredictorWeightTensorArray, true)

	local predictorTensor = Model:predict(featureVector, true)
	
	local errorVector = AqwamTensorLibrary:subtract(predictorTensor, targetTensor)
	
	local squaredErrorVector = AqwamTensorLibrary:power(errorVector, 2)
	
	local sumSquaredErrorVector = AqwamTensorLibrary:sum(squaredErrorVector, 2)
	
	local generatedVector = AqwamTensorLibrary:power(sumSquaredErrorVector, 0.5)

	Model:forwardPropagate(featureVector, true)
	
	Model:update(errorVector, true)
	
	self.TargetWeightTensorArray = TargetWeightTensorArray
	
	self.PredictorWeightTensorArray = Model:getWeightTensorArray(true)
	
	return generatedVector
	
end

function RandomNetworkDistillation:getTargetWeightTensorArray(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.TargetWeightTensorArray 
		
	else
		
		return self:deepCopyTable(self.TargetWeightTensorArray)
		
	end
	
end

function RandomNetworkDistillation:getPredictorWeightTensorArray(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.PredictorWeightTensorArray 

	else

		return self:deepCopyTable(self.PredictorWeightTensorArray)

	end
	
end

function RandomNetworkDistillation:setTargetWeightTensorArray(TargetWeightTensorArray, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.TargetWeightTensorArray = TargetWeightTensorArray

	else

		self.TargetWeightTensorArray = self:deepCopyTable(TargetWeightTensorArray)

	end
	
end

function RandomNetworkDistillation:setPredictorWeightTensorArray(PredictorWeightTensorArray, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.PredictorWeightTensorArray = PredictorWeightTensorArray

	else

		self.PredictorWeightTensorArray = self:deepCopyTable(PredictorWeightTensorArray)

	end

end

return RandomNetworkDistillation