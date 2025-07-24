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

RecurrentReinforcementLearningBaseModel = {}

RecurrentReinforcementLearningBaseModel.__index = RecurrentReinforcementLearningBaseModel

setmetatable(RecurrentReinforcementLearningBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function RecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewRecurrentReinforcementLearningBaseModel = {}

	setmetatable(NewRecurrentReinforcementLearningBaseModel, RecurrentReinforcementLearningBaseModel)

	NewRecurrentReinforcementLearningBaseModel:setName("RecurrentReinforcementLearningBaseModel")

	NewRecurrentReinforcementLearningBaseModel:setClassName("RecurrentReinforcementLearningModel")

	NewRecurrentReinforcementLearningBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewRecurrentReinforcementLearningBaseModel.Model = parameterDictionary.Model
	
	NewRecurrentReinforcementLearningBaseModel.hiddenStateTensor = parameterDictionary.hiddenStateTensor

	return NewRecurrentReinforcementLearningBaseModel

end

function RecurrentReinforcementLearningBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function RecurrentReinforcementLearningBaseModel:getDiscountFactor()

	return self.discountFactor

end

function RecurrentReinforcementLearningBaseModel:setModel(Model)

	self.Model = Model

end

function RecurrentReinforcementLearningBaseModel:getModel()

	return self.Model

end

function RecurrentReinforcementLearningBaseModel:predict(featureVector, hiddenStateVector, returnOriginalOutput)

	return self.Model:predict(featureVector, hiddenStateVector, returnOriginalOutput)

end

function RecurrentReinforcementLearningBaseModel:getClassesList()

	return self.Model:getClassesList()

end

function RecurrentReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function RecurrentReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function RecurrentReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The categorical update function is not implemented.")

	end

end

function RecurrentReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented.")

	end

end

function RecurrentReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function RecurrentReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction
	
	self.hiddenStateTensor = nil

	if (episodeUpdateFunction) then

		return episodeUpdateFunction(terminalStateValue)

	else

		error("The episode update function is not implemented.")

	end

end

function RecurrentReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function RecurrentReinforcementLearningBaseModel:reset()

	local resetFunction = self.resetFunction
	
	self.hiddenStateTensor = nil

	if (resetFunction) then 

		return resetFunction() 

	else

		error("The reset function is not implemented.")

	end

end

return RecurrentReinforcementLearningBaseModel