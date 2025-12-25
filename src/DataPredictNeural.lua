--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Axon)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Axon/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamDeepLearningLibrary = {}

local Containers = script.Containers

local Models = script.Models

local RecurrentModels = script.RecurrentModels

local ActivationBlocks = script.ActivationBlocks

local CostFunctions = script.CostFunctions

local WeightBlocks = script.WeightBlocks

local ConvolutionBlocks = script.ConvolutionBlocks

local PoolingBlocks = script.PoolingBlocks

local EncodingBlocks = script.EncodingBlocks

local DropoutBlocks = script.DropoutBlocks

local OperatorBlocks = script.OperatorBlocks

local ShapeTransformationBlocks = script.ShapeTransformationBlocks

local AttentionBlocks = script.AttentionBlocks

local PaddingBlocks = script.PaddingBlocks

local ExpansionBlocks = script.ExpansionBlocks

local HolderBlocks = script.HolderBlocks

local Optimizers = script.Optimizers

local GradientClippers = script.GradientClippers

local ValueSchedulers = script.ValueSchedulers

local Regularizers = script.Regularizers

local EligibilityTraces = script.EligibilityTraces

local Utilities = script.Utilities

local Cores = script.Cores

AqwamDeepLearningLibrary.Containers = {
	
	Sequential = require(Containers.Sequential),
	
	ComputationalGraph = require(Containers.ComputationalGraph),
	
	RNNCell = require(Containers.RecurrentNeuralNetworkCell),
	RecurrentNeuralNetworkCell = require(Containers.RecurrentNeuralNetworkCell),
	
	RNN = require(Containers.RecurrentNeuralNetwork),
	RecurrentNeuralNetwork = require(Containers.RecurrentNeuralNetwork),
	
	GRUCell = require(Containers.GatedRecurrentUnitCell),
	GatedRecurrentUnitCell = require(Containers.GatedRecurrentUnitCell),
	
	GRU = require(Containers.GatedRecurrentUnit),
	GatedRecurrentUnit = require(Containers.GatedRecurrentUnit),
	
}

AqwamDeepLearningLibrary.Models = {
	
	GAN = require(Models.GenerativeAdversarialNetwork),
	GenerativeAdversarialNetwork = require(Models.GenerativeAdversarialNetwork),
	
	WGAN = require(Models.WassersteinGenerativeAdversarialNetwork),
	WassersteinGenerativeAdversarialNetwork = require(Models.WassersteinGenerativeAdversarialNetwork),
	
	VPG = require(Models.VanillaPolicyGradient),
	VanillaPolicyGradient = require(Models.VanillaPolicyGradient),
	
	AC = require(Models.ActorCritic),
	ActorCritic = require(Models.ActorCritic),
	
	TDAC = require(Models.TemporalDifferenceActorCritic),
	TemporalDifferenceActorCritic = require(Models.TemporalDifferenceActorCritic),
	
	A2C = require(Models.AdvantageActorCritic),
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	SAC = require(Models.SoftActorCritic),
	SoftActorCritic = require(Models.SoftActorCritic),
	
	PPO = require(Models.ProximalPolicyOptimization),
	ProximalPolicyOptimization = require(Models.ProximalPolicyOptimization),
	
	PPOClip = require(Models.ProximalPolicyOptimizationClip),
	ProximalPolicyOptimizationClip = require(Models.ProximalPolicyOptimizationClip),
	
	DDPG = require(Models.DeepDeterministicPolicyGradient),
	DeepDeterministicPolicyGradient = require(Models.DeepDeterministicPolicyGradient),
	
	TD3 = require(Models.TwinDelayedDeepDeterministicPolicyGradient),
	TwinDelayedDeepDeterministicPolicyGradient = require(Models.TwinDelayedDeepDeterministicPolicyGradient),
	
	REINFORCE = require(Models.REINFORCE),
	
	MonteCarloControl = require(Models.MonteCarloControl),
	
	OffPolicyMonteCarloControl = require(Models.OffPolicyMonteCarloControl),
	
	TD = require(Models.TemporalDifference),
	TemporalDifference = require(Models.TemporalDifference),
	
	DQN = require(Models.DeepQLearning),
	DeepQLearning = require(Models.DeepQLearning),
	
	DeepSARSA = require(Models.DeepStateActionRewardStateAction),
	DeepStateActionRewardStateAction = require(Models.DeepStateActionRewardStateAction),
	
	DeepExpectedSARSA = require(Models.DeepExpectedStateActionRewardStateAction),
	DeepExpectedStateActionRewardStateAction = require(Models.DeepExpectedStateActionRewardStateAction),
	
	DeepClippedDoubleQLearning = require(Models.DeepClippedDoubleQLearning),
	
	DDQNV1 = require(Models.DeepDoubleQLearningV1),
	DeepDoubleQLearningV1 = require(Models.DeepDoubleQLearningV1),
	
	DDQNV2 = require(Models.DeepDoubleQLearningV2),
	DeepDoubleQLearningV2 = require(Models.DeepDoubleQLearningV2),
	
	DeepDoubleSARSAV1 = require(Models.DeepDoubleStateActionRewardStateActionV1),
	DeepDoubleStateActionRewardStateActionV1 = require(Models.DeepDoubleStateActionRewardStateActionV1),
	
	DeepDoubleSARSAV2 = require(Models.DeepDoubleStateActionRewardStateActionV2),
	DeepDoubleStateActionRewardStateActionV2 = require(Models.DeepDoubleStateActionRewardStateActionV2),
	
	DeepDoubleExpectedSARSAV1 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV1),
	DeepDoubleExpectedStateActionRewardStateActionV1 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV1),
	
	DeepDoubleExpectedSARSAV2 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV2),
	DeepDoubleExpectedStateActionRewardStateActionV2 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV2),
	
	RandomNetworkDistillation = require(Models.RandomNetworkDistillation),
	
	Diffusion = require(Models.Diffusion)
	
}

AqwamDeepLearningLibrary.RecurrentModels = {
	
	RecurrentVanillaPolicyGradient = require(RecurrentModels.RecurrentVanillaPolicyGradient),
	
	RecurrentActorCritic = require(RecurrentModels.RecurrentActorCritic),
	
	RecurrentAdvantageActorCritic = require(RecurrentModels.RecurrentAdvantageActorCritic),
	
	RecurrentSoftActorCritic = require(RecurrentModels.RecurrentSoftActorCritic),
	
	RecurrentProximalPolicyOptimization = require(RecurrentModels.RecurrentProximalPolicyOptimization),
	
	RecurrentProximalPolicyOptimizationClip = require(RecurrentModels.RecurrentProximalPolicyOptimizationClip),
	
	RecurrentDeepDeterministicPolicyGradient = require(RecurrentModels.RecurrentDeepDeterministicPolicyGradient),
	
	RecurrentTwinDelayedDeepDeterministicPolicyGradient = require(RecurrentModels.RecurrentTwinDelayedDeepDeterministicPolicyGradient),

	RecurrentREINFORCE = require(RecurrentModels.RecurrentREINFORCE),
	
	RecurrentMonteCarloControl = require(RecurrentModels.RecurrentMonteCarloControl),

	RecurrentOffPolicyMonteCarloControl = require(RecurrentModels.RecurrentOffPolicyMonteCarloControl),
	
	RecurrentDeepQLearning = require(RecurrentModels.RecurrentDeepQLearning),
	
	RecurrentDeepStateActionRewardStateAction = require(RecurrentModels.RecurrentDeepStateActionRewardStateAction),
	
	RecurrentDeepExpectedStateActionRewardStateAction = require(RecurrentModels.RecurrentDeepExpectedStateActionRewardStateAction),
	
	RecurrentDeepClippedDoubleQLearning = require(RecurrentModels.RecurrentDeepClippedDoubleQLearning),
	
	RecurrentDeepDoubleQLearningV1 = require(RecurrentModels.RecurrentDeepDoubleQLearningV1),
	
	RecurrentDeepDoubleQLearningV2 = require(RecurrentModels.RecurrentDeepDoubleQLearningV2),
	
	RecurrentDeepDoubleStateActionRewardStateActionV1 = require(RecurrentModels.RecurrentDeepDoubleStateActionRewardStateActionV1),
	
	RecurrentDeepDoubleStateActionRewardStateActionV2 = require(RecurrentModels.RecurrentDeepDoubleStateActionRewardStateActionV2),
	
	RecurrentDeepDoubleExpectedStateActionRewardStateActionV1 = require(RecurrentModels.RecurrentDeepDoubleExpectedStateActionRewardStateActionV1),
	
	RecurrentDeepDoubleExpectedStateActionRewardStateActionV2 = require(RecurrentModels.RecurrentDeepDoubleExpectedStateActionRewardStateActionV2),
	
}

AqwamDeepLearningLibrary.ActivationBlocks = {
	
	ReLU = require(ActivationBlocks.RectifiedLinearUnit),
	RectifiedLinearUnit = require(ActivationBlocks.RectifiedLinearUnit),
	
	LeakyReLU = require(ActivationBlocks.LeakyRectifiedLinearUnit),
	LeakyRectifiedLinearUnit = require(ActivationBlocks.LeakyRectifiedLinearUnit),
	
	SiLU = require(ActivationBlocks.SigmoidLinearUnit),
	SigmoidLinearUnit = require(ActivationBlocks.SigmoidLinearUnit),
	
	ELU = require(ActivationBlocks.ExponentialLinearUnit),
	ExponentialLinearUnit = require(ActivationBlocks.ExponentialLinearUnit),
	
	BinaryStep = require(ActivationBlocks.BinaryStep),
	
	Gaussian = require(ActivationBlocks.Gaussian),
	
	Sigmoid = require(ActivationBlocks.Sigmoid),
	
	Tanh = require(ActivationBlocks.Tanh),
	
	Mish = require(ActivationBlocks.Mish),
	
	Softmax = require(ActivationBlocks.Softmax),
	
	StableSoftmax = require(ActivationBlocks.StableSoftmax),
	
}

AqwamDeepLearningLibrary.CostFunctions = {
	
	MSE = require(CostFunctions.MeanSquaredError),
	MeanSquaredError = require(CostFunctions.MeanSquaredError),
	
	MAE = require(CostFunctions.MeanAbsoluteError),
	MeanAbsoluteError = require(CostFunctions.MeanAbsoluteError),
	
	BCE = require(CostFunctions.BinaryCrossEntropy),
	BinaryCrossEntropy = require(CostFunctions.BinaryCrossEntropy),
	
	CCE = require(CostFunctions.CategoricalCrossEntropy),
	CategoricalCrossEntropy = require(CostFunctions.CategoricalCrossEntropy),

	FocalLoss = require(CostFunctions.FocalLoss),

}

AqwamDeepLearningLibrary.WeightBlocks = {

	Linear = require(WeightBlocks.Linear),
	
	Bias = require(WeightBlocks.Bias),
	
	AutoLinear = require(WeightBlocks.AutomaticLinear),
	AutomaticLinear = require(WeightBlocks.AutomaticLinear),
	
	AutoBias = require(WeightBlocks.AutomaticBias),
	AutomaticBias = require(WeightBlocks.AutomaticBias),
	
	DataPredictLinearAndBias = require(WeightBlocks.DataPredictLinearAndBias),
	DPLAB = require(WeightBlocks.DataPredictLinearAndBias),

}

AqwamDeepLearningLibrary.ConvolutionBlocks = {
	
	Convolution1D = require(ConvolutionBlocks.Convolution1D),
	
	Convolution2D = require(ConvolutionBlocks.Convolution2D),
	
	Convolution3D = require(ConvolutionBlocks.Convolution3D),
	
	AutoConvolution1D = require(ConvolutionBlocks.AutomaticConvolution1D),
	AutomaticConvolution1D = require(ConvolutionBlocks.AutomaticConvolution1D),
	
	AutoConvolution2D = require(ConvolutionBlocks.AutomaticConvolution2D),
	AutomaticConvolution2D = require(ConvolutionBlocks.AutomaticConvolution2D),
	
	AutoConvolution3D = require(ConvolutionBlocks.AutomaticConvolution3D),
	AutomaticConvolution3D = require(ConvolutionBlocks.AutomaticConvolution3D),
	
}

AqwamDeepLearningLibrary.PoolingBlocks = {
	
	AvgPooling1D = require(PoolingBlocks.AveragePooling1D),
	AveragePooling1D = require(PoolingBlocks.AveragePooling1D),

	MaxPooling1D = require(PoolingBlocks.MaximumPooling1D),
	MaximumPooling1D = require(PoolingBlocks.MaximumPooling1D),

	MinPooling1D = require(PoolingBlocks.MinimumPooling1D),
	MinimumPooling1D = require(PoolingBlocks.MinimumPooling1D),
	
	AvgPooling2D = require(PoolingBlocks.AveragePooling2D),
	AveragePooling2D = require(PoolingBlocks.AveragePooling2D),

	MaxPooling2D = require(PoolingBlocks.MaximumPooling2D),
	MaximumPooling2D = require(PoolingBlocks.MaximumPooling2D),

	MinPooling2D = require(PoolingBlocks.MinimumPooling2D),
	MinimumPooling2D = require(PoolingBlocks.MinimumPooling2D),
	
	AvgPooling3D = require(PoolingBlocks.AveragePooling3D),
	AveragePooling3D = require(PoolingBlocks.AveragePooling3D),

	MaxPooling3D = require(PoolingBlocks.MaximumPooling3D),
	MaximumPooling3D = require(PoolingBlocks.MaximumPooling3D),

	MinPooling3D = require(PoolingBlocks.MinimumPooling3D),
	MinimumPooling3D = require(PoolingBlocks.MinimumPooling3D),

	MaxUnpooling1D = require(PoolingBlocks.MaximumUnpooling1D),
	MaximumUnpooling1D = require(PoolingBlocks.MaximumUnpooling1D),

	MaxUnpooling2D = require(PoolingBlocks.MaximumUnpooling2D),
	MaximumUnpooling2D = require(PoolingBlocks.MaximumUnpooling2D),

	MaxUnpooling3D = require(PoolingBlocks.MaximumUnpooling3D),
	MaximumUnpooling3D = require(PoolingBlocks.MaximumUnpooling3D),

}

AqwamDeepLearningLibrary.EncodingBlocks = {

	OneHotEncoding = require(EncodingBlocks.OneHotEncoding),
	
	LabelEncoding = require(EncodingBlocks.LabelEncoding),
	
	PositionalEncoding = require(EncodingBlocks.PositionalEncoding)

}

AqwamDeepLearningLibrary.DropoutBlocks = {
	
	Dropout = require(DropoutBlocks.Dropout),
	
	Dropout1D = require(DropoutBlocks.Dropout1D),
	
	Dropout2D = require(DropoutBlocks.Dropout2D),
	
	Dropout3D = require(DropoutBlocks.Dropout3D),
	
	DropoutND = require(DropoutBlocks.DropoutND),
	
}

AqwamDeepLearningLibrary.OperatorBlocks = {
	
	Add = require(OperatorBlocks.Add),
	
	Subtract = require(OperatorBlocks.Subtract),
	
	Multiply = require(OperatorBlocks.Multiply),
	
	Divide = require(OperatorBlocks.Divide),
	
	Power = require(OperatorBlocks.Power),
	
	Exponent = require(OperatorBlocks.Exponent),
	
	Logarithm = require(OperatorBlocks.Logarithm),
	
	Sum = require(OperatorBlocks.Sum),
	
	Mean = require(OperatorBlocks.Mean),
	
	StandardDeviation = require(OperatorBlocks.StandardDeviation),
	
	ZScoreNormalization = require(OperatorBlocks.ZScoreNormalization),
	
	DotProduct = require(OperatorBlocks.DotProduct),
	
	Concatenate = require(OperatorBlocks.Concatenate),
	
	Extract = require(OperatorBlocks.Extract),
	
	Clamp = require(OperatorBlocks.Clamp),
	
	Maximum = require(OperatorBlocks.Maximum),
	
	Minimum = require(OperatorBlocks.Minimum),
	
	PairwiseDistance = require(OperatorBlocks.PairwiseDistance)
	
}

AqwamDeepLearningLibrary.ShapeTransformationBlocks = {

	Transpose = require(ShapeTransformationBlocks.Transpose),

	Flatten = require(ShapeTransformationBlocks.Flatten),

	Reshape = require(ShapeTransformationBlocks.Reshape),
	
	Permute = require(ShapeTransformationBlocks.Permute),

}

AqwamDeepLearningLibrary.AttentionBlocks = {
	
	SelfAttention = require(AttentionBlocks.ScaledDotProductAttention),
	ScaledDotProductAttention = require(AttentionBlocks.ScaledDotProductAttention),
	
}

AqwamDeepLearningLibrary.PaddingBlocks = {
	
	ZeroPadding = require(PaddingBlocks.ZeroPadding),
	ZeroPad = require(PaddingBlocks.ZeroPadding),
	
	CircularPadding = require(PaddingBlocks.CircularPadding),
	CircularPad = require(PaddingBlocks.CircularPadding),
	
	ConstantPadding = require(PaddingBlocks.ConstantPadding),
	ConstantPad = require(PaddingBlocks.ConstantPadding),
	
	ReflectionPadding = require(PaddingBlocks.ReflectionPadding),
	ReflectionPad = require(PaddingBlocks.ReflectionPadding),
	
	ReplicationPadding = require(PaddingBlocks.ReplicationPadding),
	ReplicationPad = require(PaddingBlocks.ReplicationPadding),
	
}

AqwamDeepLearningLibrary.ExpansionBlocks = {
	
	ExpandDimensionSizes = require(ExpansionBlocks.ExpandDimensionSizes),
	
	ExpandNumberOfDimensions = require(ExpansionBlocks.ExpandNumberOfDimensions),
	
}

AqwamDeepLearningLibrary.HolderBlocks = {
	
	InputHolder = require(HolderBlocks.InputHolder),
	
	VariableHolder = require(HolderBlocks.VariableHolder),
	
	NullaryFunctionHolder = require(HolderBlocks.NullaryFunctionHolder),
	
}

AqwamDeepLearningLibrary.Optimizers = {

	AdaDelta = require(Optimizers.AdaptiveDelta),
	AdaptiveDelta = require(Optimizers.AdaptiveDelta),
	
	AdaFactor = require(Optimizers.AdaptiveFactor),
	AdaptiveFactor = require(Optimizers.AdaptiveFactor),
	
	AdaGrad = require(Optimizers.AdaptiveGradient),
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),
	
	Adam = require(Optimizers.AdaptiveMomentEstimation),
	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),
	
	AdaMax = require(Optimizers.AdaptiveMomentEstimationMaximum),
	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),
	
	AdamW = require(Optimizers.AdaptiveMomentEstimationWeightDecay),
	AdaptiveMomentEstimationWeightDecay = require(Optimizers.AdaptiveMomentEstimationWeightDecay),
	
	Gravity = require(Optimizers.Gravity),
	
	Momentum = require(Optimizers.Momentum),
	
	NAdam = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	
	RAdam = require(Optimizers.RectifiedAdaptiveMomentEstimation),
	RectifiedAdaptiveMomentEstimation = require(Optimizers.RectifiedAdaptiveMomentEstimation),
	
	RProp = require(Optimizers.ResilientBackwardPropagation),
	ResilientBackwardPropagation = require(Optimizers.ResilientBackwardPropagation),
	
	RMSProp  = require(Optimizers.RootMeanSquarePropagation),
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),
	
}

AqwamDeepLearningLibrary.ValueSchedulers = {

	Chained = require(ValueSchedulers.Chained),

	Constant = require(ValueSchedulers.Constant),

	CosineAnnealing = require(ValueSchedulers.CosineAnnealing),

	Exponential = require(ValueSchedulers.Exponential),

	InverseSquareRoot = require(ValueSchedulers.InverseSquareRoot),

	InverseTime = require(ValueSchedulers.InverseTime),

	Linear = require(ValueSchedulers.Linear),

	MultipleStep = require(ValueSchedulers.MultipleStep),

	Multiplicative = require(ValueSchedulers.Multiplicative),

	Polynomial = require(ValueSchedulers.Polynomial),

	Sequential = require(ValueSchedulers.Sequential),

	Step = require(ValueSchedulers.Step),

}

AqwamDeepLearningLibrary.Regularizers = {

	L1 = require(Regularizers.Lasso),
	Lasso = require(Regularizers.Lasso),

	L2 = require(Regularizers.Ridge),
	Ridge = require(Regularizers.Ridge),

	L1L2 = require(Regularizers.ElasticNet),
	ElasticNet = require(Regularizers.ElasticNet),

}

AqwamDeepLearningLibrary.GradientClippers = {

	ClipValue = require(GradientClippers.ClipValue),

	ClipNormalization = require(GradientClippers.ClipNormalization),

}

AqwamDeepLearningLibrary.EligibilityTraces = {

	AccumulatingTrace = require(EligibilityTraces.AccumulatingTrace),

	ReplacingTrace = require(EligibilityTraces.ReplacingTrace),

	DutchTrace = require(EligibilityTraces.DutchTrace),

}

AqwamDeepLearningLibrary.Utilities = {
	
	IterativeTrainingWrapper = require(Utilities.IterativeTrainingWrapper),
	
	TensorToClassConverter = require(Utilities.TensorToClassConverter),
	
	Tokenizer = require(Utilities.Tokenizer)
	
}

AqwamDeepLearningLibrary.Cores = {
	
	BaseFunctionBlock = require(Cores.BaseFunctionBlock),
	
	BaseInstance = require(Cores.BaseInstance),
	
	AutomaticDifferentiationTensor = require(Cores.AutomaticDifferentiationTensor),
	
	SymbolicDifferentiationTensor = require(Cores.SymbolicDifferentiationTensor),
	
}

return AqwamDeepLearningLibrary
