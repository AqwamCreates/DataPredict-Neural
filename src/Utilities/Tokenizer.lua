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

local Tokenizer = {}

Tokenizer.__index = Tokenizer

setmetatable(Tokenizer, BaseInstance)

function Tokenizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTokenizer = {}
	
	setmetatable(NewTokenizer, Tokenizer)
	
	NewTokenizer:setName("Tokenizer")

	NewTokenizer:setClassName("Tokenizer")
		
	NewTokenizer.tokenizedItemArray = parameterDictionary.tokenizedItemArray or {}
	
	return NewTokenizer
	
end

function Tokenizer:addItem(item)
	
	if table.find(self.tokenizedItemArray, item) then return nil end

	table.insert(self.tokenizedItemArray, item)
	
end

function Tokenizer:addAllItems(itemArray)
	
	repeat
		
		self:addItem(itemArray[1])

		table.remove(itemArray, 1)

	until (#itemArray <= 0)
	
end

function Tokenizer:convertTokenToItem(tokenNumber)
	
	return self.tokenizedItemArray[tokenNumber]
	
end

function Tokenizer:convertItemToToken(item)

	return table.find(self.tokenizedItemArray, item)

end

function Tokenizer:getTokenizedItemArray()
	
	return self.tokenizedItemArray
	
end

function Tokenizer:setTokenizedItemArray(tokenizedItemArray)

	self.tokenizedItemArray = tokenizedItemArray

end

return Tokenizer
