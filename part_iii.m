% Part III


digitDatasetPath = fullfile('scenes_lazebnik');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true, 'LabelSource', 'foldernames');
numTrainingFiles = 100;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');
net = alexnet;
net.Layers
alex_layers = net.Layers(1:22);
layers = [
    alex_layers
    fullyConnectedLayer(8)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',1,...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)