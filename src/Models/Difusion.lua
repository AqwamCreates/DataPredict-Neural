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

DiffusionModel = {}

DiffusionModel.__index = DiffusionModel

setmetatable(DiffusionModel, BaseModel)

local defaultNumberOfDiffusionStep = 300

local defaultNumberOfSamplingStep = 300

local defaultSampler = "DDPM"

local defaultNoiseScheduler = "Linear"

local defaultInitialNoiseValue = 0.0001

local defaultFinalNoiseValue = 1

local noiseSchedulerFunctionList = {
	
	["Linear"] = function(numberOfSteps, initialNoiseValue, finalNoiseValue)
		
		local betaArray = {}
		
		local noiseValueDifference = finalNoiseValue - initialNoiseValue
		
		for t = 1, numberOfSteps, 1 do
			
			local ratio = (t - 1) / (numberOfSteps - 1)
			
			betaArray[t] = initialNoiseValue + (ratio * noiseValueDifference)
				
		end
		
		return betaArray
		
	end,
	
}

local samplerFunctionList = {
	
	["Euler"] = function(Model, featureTensor, alpha, cumulativeAlpha, currentStandardDeviation, nextStandardDeviation)
		
		local ratio = nextStandardDeviation / currentStandardDeviation
		
		local complementRatio =  1 - ratio
		
		local generatedFeatureTensor = Model:forwardPropagate(featureTensor)
		
		local previousFeatureTensorPart1 = AqwamTensorLibrary:multiply(featureTensor, ratio)
		
		local previousFeatureTensorPart2 = AqwamTensorLibrary:multiply(generatedFeatureTensor, complementRatio)
		
		local previousFeatureTensor = AqwamTensorLibrary:add(previousFeatureTensorPart1, previousFeatureTensorPart2)
		
		return previousFeatureTensor
		
	end,
	
	["DDPM"] = function(Model, featureTensor, alpha, cumulativeAlpha, currentStandardDeviation, nextStandardDeviation)
		
		local inverseSquareRootAlpha = 1 / math.sqrt(alpha)
		
		local ratio = (1 - alpha) / math.sqrt((1 - cumulativeAlpha))
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureTensor)
		
		local gaussianNoise = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)
		
		local generatedFeatureTensor = Model:forwardPropagate(featureTensor)
		
		local previousFeatureTensorPart1 = AqwamTensorLibrary:multiply(generatedFeatureTensor, ratio)
		
		local previousFeatureTensorPart2 = AqwamTensorLibrary:subtract(featureTensor, previousFeatureTensorPart1)
		
		local previousFeatureTensorPart3 = AqwamTensorLibrary:multiply(previousFeatureTensorPart2, inverseSquareRootAlpha)
		
		local previousFeatureTensorPart4 = AqwamTensorLibrary:multiply(gaussianNoise, currentStandardDeviation)
		
		local previousFeatureTensor = AqwamTensorLibrary:add(previousFeatureTensorPart3, previousFeatureTensorPart4)
		
		return previousFeatureTensor
		
	end,
	
}

function DiffusionModel.new(parameterDictionary)
	
	local NewDiffusionModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewDiffusionModel, DiffusionModel)
	
	NewDiffusionModel:setName("Diffusion")
	
	local Model = parameterDictionary.Model
	
	if (not Model) then error("No model.") end
	
	NewDiffusionModel.Model = Model
	
	NewDiffusionModel.numberOfDiffusionStep = parameterDictionary.numberOfDiffusionStep or defaultNumberOfDiffusionStep
	
	NewDiffusionModel.numberOfSamplingStep = parameterDictionary.numberOfSamplingStep or defaultNumberOfSamplingStep
	
	NewDiffusionModel.sampler = parameterDictionary.sampler or defaultSampler
	
	NewDiffusionModel.noiseScheduler = parameterDictionary.noiseScheduler or defaultNoiseScheduler
	
	NewDiffusionModel.initialNoiseValue = parameterDictionary.initialNoiseValue or defaultInitialNoiseValue
	
	NewDiffusionModel.finalNoiseValue = parameterDictionary.finalNoiseValue or defaultFinalNoiseValue
	
	return NewDiffusionModel
	
end

function DiffusionModel:diffuse(featureTensor, alpha)
	
	local Model = self.Model
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(featureTensor)
	
	local gaussianNoiseTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)
	
	local squareRootAlpha = math.sqrt(alpha)
	
	local squareRootComplementAlpha = math.sqrt(1 - alpha)
	
	local noisyFeatureTensorPart1 = AqwamTensorLibrary:multiply(featureTensor, squareRootAlpha)
	
	local noisyFeatureTensorPart2 = AqwamTensorLibrary:multiply(gaussianNoiseTensor, squareRootComplementAlpha)
	
	local noisyFeatureTensor = AqwamTensorLibrary:add(noisyFeatureTensorPart1, noisyFeatureTensorPart2)
	
	local lossTensor = AqwamTensorLibrary:subtract(featureTensor, noisyFeatureTensor)
	
	Model:forwardPropagate(featureTensor, true)
	
	Model:update(lossTensor, true)
	
end

function DiffusionModel:train(featureTensor)
	
	local betaArray = noiseSchedulerFunctionList[self.noiseScheduler](self.numberOfDiffusionStep, self.initialNoiseValue, self.finalNoiseValue)
	
	local cumulativeAlpha = 1
	
	for _, beta in ipairs(betaArray) do
		
		local alpha = 1 - beta
		
		cumulativeAlpha = cumulativeAlpha * alpha
	
		self:diffuse(featureTensor, cumulativeAlpha) 
		
	end
	
end

function DiffusionModel:sample(featureTensor, alpha, cumulativeAlpha, currentStandardDeviation, nextStandardDeviation)
	
	return samplerFunctionList[self.sampler](self.Model, featureTensor, alpha, cumulativeAlpha, currentStandardDeviation, nextStandardDeviation)
	
end

function DiffusionModel:generate(featureTensor)
	
	local betaArray = noiseSchedulerFunctionList[self.noiseScheduler](self.numberOfDiffusionStep, self.initialNoiseValue, self.finalNoiseValue)
	
	local cumulativeAlpha = 1
	
	local numberOfBetas = #betaArray
	
	for i = (#betaArray - 1), 2, -1 do 
		
		local currentBeta = betaArray[i]
		
		local nextBeta = betaArray[i + 1]
		
		local alpha = 1 - currentBeta

		local currentStandardDeviation = math.sqrt(currentBeta)
		
		local nextStandardDeviation = math.sqrt(nextBeta)
		
		cumulativeAlpha = cumulativeAlpha * alpha
		
		featureTensor = self:sample(featureTensor, alpha, cumulativeAlpha, currentStandardDeviation, nextStandardDeviation)
		
	end
	
	return featureTensor
	
end

return DiffusionModel
