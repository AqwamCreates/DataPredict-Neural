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

DualRecurrentReinforcementLearningBaseModel = {}

DualRecurrentReinforcementLearningBaseModel.__index = DualRecurrentReinforcementLearningBaseModel

setmetatable(DualRecurrentReinforcementLearningBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function DualRecurrentReinforcementLearningBaseModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewDualRecurrentReinforcementLearningBaseModel = {}

	setmetatable(NewDualRecurrentReinforcementLearningBaseModel, DualRecurrentReinforcementLearningBaseModel)

	NewDualRecurrentReinforcementLearningBaseModel:setName("DualRecurrentReinforcementLearningBaseModel")

	NewDualRecurrentReinforcementLearningBaseModel:setClassName("DualRecurrentReinforcementLearningModel")

	NewDualRecurrentReinforcementLearningBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewDualRecurrentReinforcementLearningBaseModel.Model = parameterDictionary.Model

	NewDualRecurrentReinforcementLearningBaseModel.hiddenStateTensorArray = {parameterDictionary.hiddenStateTensor1, parameterDictionary.hiddenStateTensor2}

	return NewDualRecurrentReinforcementLearningBaseModel

end

function DualRecurrentReinforcementLearningBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function DualRecurrentReinforcementLearningBaseModel:getDiscountFactor()

	return self.discountFactor

end

function DualRecurrentReinforcementLearningBaseModel:setModel(Model)

	self.Model = Model

end

function DualRecurrentReinforcementLearningBaseModel:getModel()

	return self.Model

end

function DualRecurrentReinforcementLearningBaseModel:predict(featureTensor, hiddenStateTensor, returnOriginalOutput)

	return self.Model:predict(featureTensor, hiddenStateTensor, returnOriginalOutput)

end

function DualRecurrentReinforcementLearningBaseModel:getClassesList()

	return self.Model:getClassesList()

end

function DualRecurrentReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function DualRecurrentReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function DualRecurrentReinforcementLearningBaseModel:categoricalUpdate(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureTensor, previousAction, rewardValue, currentFeatureTensor, currentAction, terminalStateValue)

	else

		error("The categorical update function is not implemented.")

	end

end

function DualRecurrentReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then

		return diagonalGaussianUpdateFunction(previousFeatureTensor, previousActionMeanTensor, previousActionStandardDeviationTensor, previousActionNoiseTensor, rewardValue, currentFeatureTensor, currentActionMeanTensor, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented.")

	end

end

function DualRecurrentReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function DualRecurrentReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction
	
	self.hiddenStateTensorArray = {}

	if (episodeUpdateFunction) then

		return episodeUpdateFunction(terminalStateValue)

	else

		error("The episode update function is not implemented.")

	end

end

function DualRecurrentReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function DualRecurrentReinforcementLearningBaseModel:reset()

	local resetFunction = self.resetFunction
	
	self.hiddenStateTensorArray = {}

	if (resetFunction) then 

		return resetFunction() 

	else

		error("The reset function is not implemented.")

	end

end

return DualRecurrentReinforcementLearningBaseModel
