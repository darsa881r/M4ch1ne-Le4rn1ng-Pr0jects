close all
clearvars
clc

% The MNIST image extraction code is taken from
% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
%https://www.mathworks.com/help/nnet/examples/create-simple-deep-learning-network-for-classification.html
%https://www.mathworks.com/help/nnet/examples/visualize-features-of-a-convolutional-neural-network.html

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images = images';
% table = [images labels];
% csvwrite('table.dat',table);
% ttds = tabularTextDatastore('table.dat');

%% Storing all the extraced images into a 3D array.

for i = 1:60000   
    I(:,:,1,i) = reshape(images(i,:),[28,28]);
end
%clear images

%% Showing some of the random images form the stored dataset
rng(0)
figure; 
perm = randperm(60000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(I(:,:,1,perm(i)));
end
%pause;
%% counting the number of examples for each Labels and defining size of image

for i=1:10
    lcount(i,1) = length(find(labels==(i-1)));
end

image_size = size(I(:,:,1));
image_size = [image_size 1]; % 1 = if one channer, 3 = if rgb channels
Total_N = length(I);

%% Splitting Training Data set into Validation set and Training Set and test set

p_test = 0.15;
p_valid = 0.10;
p_train = 1-p_test-p_valid;

n_train = floor(Total_N*p_train);
n_valid = floor(Total_N*p_valid);
n_test = Total_N-n_train-n_valid;

index_valid = randperm(Total_N,n_valid);
index_test = randperm(Total_N,n_test);

valid_set = I(:,:,1,index_valid);
valid_labels = categorical(labels(index_valid));

test_set = I(:,:,1,index_test);
test_labels = categorical(labels(index_test));

train_set=I;
train_labels=categorical(labels);

index_delete = [index_valid index_test];

train_set(:,:,:,index_delete)=[]; 
train_labels(index_delete)=[];


%clear labels I 

%% Converting to matrix for matlab neural network function

train_cell = {train_set, train_labels};
valid_cell = {valid_set, valid_labels};
test_cell = {test_set, test_labels};

%% Setting up the NN Layers

layers = [
    imageInputLayer(image_size)
    
    convolution2dLayer(6,10,'Stride',[1 1],'Padding','same',...
    'WeightLearnRateFactor',1,'BiasLearnRateFactor',1,...
    'WeightL2Factor',1,'BiasL2Factor',1)
    
    batchNormalizationLayer % creates default batch normalization layer
    
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,10,'Stride',[1 1],'Padding','same',...
    'WeightLearnRateFactor',1,'BiasLearnRateFactor',1,...
    'WeightL2Factor',1,'BiasL2Factor',1)
    
    batchNormalizationLayer % creates default batch normalization layer
    
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)  
    
    convolution2dLayer(3,10,'Stride',[1 1],'Padding','same',...
    'WeightLearnRateFactor',1,'BiasLearnRateFactor',1,...
    'WeightL2Factor',1,'BiasL2Factor',1)
    
    batchNormalizationLayer
   
    
    reluLayer  
    
    fullyConnectedLayer(10)
    
    softmaxLayer
    
    classificationLayer];


% % Manually initialize the weights from a Gaussian distribution with standard deviation of 0.0001.
% 
% layer.Weights = randn([4 4 1 3]) * 0.0001;
% % Initialize the biases from a Gaussian distribution with a mean of 1 and a standard deviation of 0.00001.
% layer.Bias = randn([1 1 32])*0.00001 + 1;
% lgraph = layerGraph(net.Layers);
% figure
% plot(lgraph)
analyzeNetwork(layers)

%% Specify Training Options and Run training network 'ExecutionEnvironment' ,'parallel''ValidationData',valid_cell,'ValidationFrequency',20,

% options2 = trainingOptions('sgdm','MaxEpochs',4,  ...
%     'InitialLearnRate',0.01,'ValidationData',valid_cell, ...
%     'ValidationFrequency',20,'Verbose',false,'Shuffle','every-epoch','MiniBatchSize',128, ...
%     'Plots','training-progress','L2Regularization',0.001);
% 
% [net2,traininfo2]  = trainNetwork(train_set,train_labels,layers,options2);


options = trainingOptions('sgdm','MaxEpochs',4,  ...
    'InitialLearnRate',0.01, ...
    'Verbose',false,'Shuffle','every-epoch','MiniBatchSize',128, ...
    'Plots','training-progress','L2Regularization',0.001);
[net,traininfo]  = trainNetwork(I,categorical(labels),layers,options);


%% Visualization of ConvNet featureewss weuiightd

channels = 1:10;
layer_number = 2;
I1 = deepDreamImage(net,layer_number,channels,...
    'PyramidLevels',1);

figure
montage(I1) %'ThumbnailSize',[30,30]
name = net.Layers(layer_number).Name;
title(['Layer ',name,' Features'])

% figure
% for i = 1:10
%     subplot(3,4,i)
%     imshow(I1(:,:,:,i),[])
% end


channels = 1:10;
layer_number = 6;
I2 = deepDreamImage(net,layer_number,channels, ...
    'PyramidLevels',1);

figure
montage(I2)
name = net.Layers(layer_number).Name;
title(['Layer ',name,' Features'])

% figure
% for i = 1:10
%     subplot(3,4,i)
%     imshow(I2(:,:,:,i),[])
% end

channels = 1:10;
layer_number = 10;
I3 = deepDreamImage(net,layer_number,channels, ...
    'PyramidLevels',1);

figure
montage(I3)
name = net.Layers(layer_number).Name;
title(['Layer ',name,' Features'])

% figure
% for i = 1:10
%     subplot(3,4,i)
%     imshow(I3(:,:,:,i),[])
% end
%% visualizing fully conected network

channels = 1:10;
layer_number = 13;

IF1 = deepDreamImage(net,layer_number,channels, ...
    'Verbose',false, ...
    'NumIterations',50);

figure
montage(IF1)
name = net.Layers(layer_number).Name;
title(['Layer ',name,' Features'])

% layer_number = 13;
% channels = 1:10;
% net.Layers(end).ClassNames(channels)
% 
% IF2 = deepDreamImage(net,layer_number,channels, ...
%     'Verbose',false, ...
%     'NumIterations',50);
% 
% figure
% montage(IF2)
% name = net.Layers(layer_number).Name;
% title(['Layer ',name,' Features'])

%% Classify Train Images and Compute Accuracy

tic
train_YPred = classify(net2,train_set);
train_accuracy = sum(train_YPred == train_labels)/numel(train_labels)
toc
figure
plotconfusion(train_labels,train_YPred)

%% Classify Test Images and Compute Accuracy

tic
test_YPred = classify(net2,test_set);
test_accuracy = sum(test_YPred == test_labels)/numel(test_labels)
toc
figure
plotconfusion(test_labels,test_YPred)

