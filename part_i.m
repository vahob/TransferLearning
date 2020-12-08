% Part I
% 1
digitDatasetPath = fullfile('scenes_lazebnik');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true, 'LabelSource', 'foldernames');
numTrainingFiles = 100;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

% 2
layers = [
    imageInputLayer([227 227 3])
    convolution2dLayer(11, 50)
    reluLayer
    maxPooling2dLayer(3, 'Stride', 1)
    convolution2dLayer(5, 60)
    reluLayer
    maxPooling2dLayer(3, 'Stride', 2)
    fullyConnectedLayer(8)
    softmaxLayer
    classificationLayer];
% 3
options = trainingOptions('sgdm', ...
    'MaxEpochs',1,...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 4
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)