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

DualRecurrentReinforcementLearningActorCriticBaseModel = {}

DualRecurrentReinforcementLearningActorCriticBaseModel.__index = DualRecurrentReinforcementLearningActorCriticBaseModel

setmetatable(DualRecurrentReinforcementLearningActorCriticBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function DualRecurrentReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDualRecurrentReinforcementLearningActorCriticBaseModel = {}

	setmetatable(NewDualRecurrentReinforcementLearningActorCriticBaseModel, DualRecurrentReinforcementLearningActorCriticBaseModel)

	NewDualRecurrentReinforcementLearningActorCriticBaseModel:setName("DualRecurrentReinforcementLearningActorCriticBaseModel")

	NewDualRecurrentReinforcementLearningActorCriticBaseModel:setClassName("DualRecurrentReinforcementLearningActorCriticModel")

	NewDualRecurrentReinforcementLearningActorCriticBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewDualRecurrentReinforcementLearningActorCriticBaseModel.ActorModel = parameterDictionary.ActorModel

	NewDualRecurrentReinforcementLearningActorCriticBaseModel.CriticModel = parameterDictionary.CriticModel

	NewDualRecurrentReinforcementLearningActorCriticBaseModel.actorHiddenStateTensorArray = {parameterDictionary.actorHiddenStateTensor1, parameterDictionary.actorHiddenStateTensor2}

	NewDualRecurrentReinforcementLearningActorCriticBaseModel.criticHiddenStateValueArray = {parameterDictionary.criticHiddenStateValue1, parameterDictionary.criticHiddenStateValue2}

	return NewDualRecurrentReinforcementLearningActorCriticBaseModel

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:getDiscountFactor()

	return self.discountFactor

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setActorModel(ActorModel)

	self.ActorModel = ActorModel

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setCriticModel(CriticModel)

	self.CriticModel = CriticModel

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:getActorModel()

	return self.ActorModel

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:getCriticModel()

	return self.CriticModel

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:predict(featureVector, hiddenStateTensor, returnOriginalOutput)

	return self.ActorModel:predict(featureVector, hiddenStateTensor, returnOriginalOutput)

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:getClassesList()

	return self.ActorModel:getClassesList()

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The categorical update function is not implemented!")

	end

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented!")

	end

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction

	self.actorHiddenStateTensorArray = {}

	self.criticHiddenStateValueArray = {}

	if (episodeUpdateFunction) then

		return episodeUpdateFunction(terminalStateValue)

	else

		error("The episode update function is not implemented!")

	end

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function DualRecurrentReinforcementLearningActorCriticBaseModel:reset()

	local resetFunction = self.resetFunction

	self.actorHiddenStateTensorArray = {}

	self.criticHiddenStateValueArray = {}

	if (resetFunction) then 

		return resetFunction() 

	else

		error("The reset function is not implemented!")

	end

end

return DualRecurrentReinforcementLearningActorCriticBaseModel