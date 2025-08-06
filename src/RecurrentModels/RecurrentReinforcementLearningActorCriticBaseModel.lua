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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

RecurrentReinforcementLearningActorCriticBaseModel = {}

RecurrentReinforcementLearningActorCriticBaseModel.__index = RecurrentReinforcementLearningActorCriticBaseModel

setmetatable(RecurrentReinforcementLearningActorCriticBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function RecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentReinforcementLearningActorCriticBaseModel = {}

	setmetatable(NewRecurrentReinforcementLearningActorCriticBaseModel, RecurrentReinforcementLearningActorCriticBaseModel)

	NewRecurrentReinforcementLearningActorCriticBaseModel:setName("RecurrentReinforcementLearningActorCriticBaseModel")

	NewRecurrentReinforcementLearningActorCriticBaseModel:setClassName("RecurrentReinforcementLearningActorCriticModel")

	NewRecurrentReinforcementLearningActorCriticBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewRecurrentReinforcementLearningActorCriticBaseModel.ActorModel = parameterDictionary.ActorModel

	NewRecurrentReinforcementLearningActorCriticBaseModel.CriticModel = parameterDictionary.CriticModel
	
	NewRecurrentReinforcementLearningActorCriticBaseModel.actorHiddenStateTensor = parameterDictionary.actorHiddenStateTensor
	
	NewRecurrentReinforcementLearningActorCriticBaseModel.criticHiddenStateValue = parameterDictionary.criticHiddenStateValue

	return NewRecurrentReinforcementLearningActorCriticBaseModel

end

function RecurrentReinforcementLearningActorCriticBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function RecurrentReinforcementLearningActorCriticBaseModel:getDiscountFactor()

	return self.discountFactor

end

function RecurrentReinforcementLearningActorCriticBaseModel:setActorModel(ActorModel)

	self.ActorModel = ActorModel

end

function RecurrentReinforcementLearningActorCriticBaseModel:setCriticModel(CriticModel)

	self.CriticModel = CriticModel

end

function RecurrentReinforcementLearningActorCriticBaseModel:getActorModel()

	return self.ActorModel

end

function RecurrentReinforcementLearningActorCriticBaseModel:getCriticModel()

	return self.CriticModel

end

function RecurrentReinforcementLearningActorCriticBaseModel:predict(featureVector, hiddenStateTensor, returnOriginalOutput)

	return self.ActorModel:predict(featureVector, hiddenStateTensor, returnOriginalOutput)

end

function RecurrentReinforcementLearningActorCriticBaseModel:getActionsList()

	return self.ActorModel:getClassesList()

end

function RecurrentReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function RecurrentReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function RecurrentReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The categorical update function is not implemented!")

	end

end

function RecurrentReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented!")

	end

end

function RecurrentReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function RecurrentReinforcementLearningActorCriticBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction
	
	self.actorHiddenStateTensor = nil

	self.criticHiddenStateValue = nil

	if (episodeUpdateFunction) then

		return episodeUpdateFunction(terminalStateValue)

	else

		error("The episode update function is not implemented!")

	end

end

function RecurrentReinforcementLearningActorCriticBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function RecurrentReinforcementLearningActorCriticBaseModel:reset()

	local resetFunction = self.resetFunction
	
	self.actorHiddenStateTensor = nil
	
	self.criticHiddenStateValue = nil

	if (resetFunction) then 

		return resetFunction() 

	else

		error("The reset function is not implemented!")

	end

end

return RecurrentReinforcementLearningActorCriticBaseModel
