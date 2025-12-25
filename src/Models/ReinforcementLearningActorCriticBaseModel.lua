--[[

	--------------------------------------------------------------------

	Aqwam's Machine,  And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local ReinforcementLearningActorCriticBaseModel = {}

ReinforcementLearningActorCriticBaseModel.__index = ReinforcementLearningActorCriticBaseModel

setmetatable(ReinforcementLearningActorCriticBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewReinforcementLearningActorCriticBaseModel = {}

	setmetatable(NewReinforcementLearningActorCriticBaseModel, ReinforcementLearningActorCriticBaseModel)

	NewReinforcementLearningActorCriticBaseModel:setName("ReinforcementLearningActorCriticBaseModel")

	NewReinforcementLearningActorCriticBaseModel:setClassName("ReinforcementLearningActorCriticModel")

	NewReinforcementLearningActorCriticBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewReinforcementLearningActorCriticBaseModel.ActorModel = parameterDictionary.ActorModel

	NewReinforcementLearningActorCriticBaseModel.CriticModel = parameterDictionary.CriticModel

	return NewReinforcementLearningActorCriticBaseModel

end

function ReinforcementLearningActorCriticBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function ReinforcementLearningActorCriticBaseModel:getDiscountFactor()

	return self.discountFactor

end

function ReinforcementLearningActorCriticBaseModel:setActorModel(ActorModel)

	self.ActorModel = ActorModel

end

function ReinforcementLearningActorCriticBaseModel:setCriticModel(CriticModel)

	self.CriticModel = CriticModel

end

function ReinforcementLearningActorCriticBaseModel:getActorModel()

	return self.ActorModel

end

function ReinforcementLearningActorCriticBaseModel:getCriticModel()

	return self.CriticModel

end

function ReinforcementLearningActorCriticBaseModel:setModelParametersArray(ModelParametersArray, doNotCopy)

	self.ActorModel:setModelParameters(ModelParametersArray[1], doNotCopy)

	self.CriticModel:setModelParameters(ModelParametersArray[2], doNotCopy)

end

function ReinforcementLearningActorCriticBaseModel:getModelParametersArray(doNotCopy)

	local ActorModelParameters = self.ActorModel:getModelParameters(doNotCopy)

	local CriticModelParameters = self.CriticModel:getModelParameters(doNotCopy)

	return {ActorModelParameters, CriticModelParameters}

end

function ReinforcementLearningActorCriticBaseModel:predict(featureTensor, returnOriginalOutput)

	return self.ActorModel:predict(featureTensor, returnOriginalOutput)

end

function ReinforcementLearningActorCriticBaseModel:setActionsList(ActionsList)

	self.ActorModel:setClassesList(ActionsList)

end

function ReinforcementLearningActorCriticBaseModel:getActionsList()

	return self.ActorModel:getClassesList()

end

function ReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

	return self.categoricalUpdateFunction(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

end

function ReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

	return self.diagonalGaussianUpdateFunction(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

end

function ReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:episodeUpdate(terminalStateValue)

	return self.episodeUpdateFunction(terminalStateValue)

end

function ReinforcementLearningActorCriticBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningActorCriticBaseModel:reset()

	return self.resetFunction() 

end

return ReinforcementLearningActorCriticBaseModel
