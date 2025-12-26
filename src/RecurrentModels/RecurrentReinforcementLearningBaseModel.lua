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

function RecurrentReinforcementLearningBaseModel:predict(featureTensor, hiddenStateTensor, returnOriginalOutput)

	return self.Model:predict(featureTensor, hiddenStateTensor, returnOriginalOutput)

end

function RecurrentReinforcementLearningBaseModel:getActionsList()

	return self.Model:getClassesList()

end

function RecurrentReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function RecurrentReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function RecurrentReinforcementLearningBaseModel:categoricalUpdate(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

	return self.categoricalUpdateFunction(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

end

function RecurrentReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

	return self.diagonalGaussianUpdateFunction(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

end

function RecurrentReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function RecurrentReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)
	
	self.hiddenStateTensor = nil

	return self.episodeUpdateFunction(terminalStateValue)

end

function RecurrentReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function RecurrentReinforcementLearningBaseModel:reset()
	
	self.hiddenStateTensor = nil

	return self.resetFunction() 

end

return RecurrentReinforcementLearningBaseModel
