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

%analyzeNetwork(layers)

pause;
%% Specify Training Options and Run training network 'ExecutionEnvironment' ,'parallel''ValidationData',valid_cell,'ValidationFrequency',20,

% options2 = trainingOptions('sgdm','MaxEpochs',4,  ...
%     'InitialLearnRate',0.01,'ValidationData',valid_cell, ...
%     'ValidationFrequency',20,'Verbose',false,'Shuffle','every-epoch','MiniBatchSize',128, ...
%     'Plots','training-progress','L2Regularization',0.001,'Momentum',0.95);
% 
% [net2,traininfo2]  = trainNetwork(train_set,train_labels,layers,options2);


options = trainingOptions('sgdm','MaxEpochs',2,  ...
    'InitialLearnRate',0.01, ...
    'Verbose',false,'Shuffle','every-epoch','MiniBatchSize',128, ...
    'Plots','training-progress','L2Regularization',0.001,'Momentum',0.95);
[net,traininfo]  = trainNetwork(I,categorical(labels),layers,options);

pause;
%% Visualization of ConvNet featureewss weuiightd

% channels = 1:10;
% layer_number = 2;
% I1 = deepDreamImage(net,layer_number,channels,...
%     'PyramidLevels',1);
% 
% figure
% montage(I1) %'ThumbnailSize',[30,30]
% name = net.Layers(layer_number).Name;
% title(['Layer ',name,' Features'])
% 
% % figure
% % for i = 1:10
% %     subplot(3,4,i)
% %     imshow(I1(:,:,:,i),[])
% % end
% 
% 
% channels = 1:10;
% layer_number = 6;
% I2 = deepDreamImage(net,layer_number,channels, ...
%     'PyramidLevels',1);
% 
% figure
% montage(I2)
% name = net.Layers(layer_number).Name;
% title(['Layer ',name,' Features'])
% 
% % figure
% % for i = 1:10
% %     subplot(3,4,i)
% %     imshow(I2(:,:,:,i),[])
% % end
% 
% channels = 1:10;
% layer_number = 10;
% I3 = deepDreamImage(net,layer_number,channels, ...
%     'PyramidLevels',1);
% 
% figure
% montage(I3)
% name = net.Layers(layer_number).Name;
% title(['Layer ',name,' Features'])

% figure
% for i = 1:10
%     subplot(3,4,i)
%     imshow(I3(:,:,:,i),[])
% end
%% visualizing fully conected network

% channels = 1:10;
% layer_number = 13;
% 
% IF1 = deepDreamImage(net,layer_number,channels, ...
%     'Verbose',false, ...
%     'NumIterations',50);
% 
% figure
% montage(IF1)
% name = net.Layers(layer_number).Name;
% title(['Layer ',name,' Features'])
% 
% % layer_number = 13;
% % channels = 1:10;
% % net.Layers(end).ClassNames(channels)
% % 
% % IF2 = deepDreamImage(net,layer_number,channels, ...
% %     'Verbose',false, ...
% %     'NumIterations',50);
% % 
% % figure
% % montage(IF2)
% % name = net.Layers(layer_number).Name;
% % title(['Layer ',name,' Features'])

%% Classify Train Images and Compute Accuracy

tic
train_YPred = classify(net,train_set);
train_accuracy = sum(train_YPred == train_labels)/numel(train_labels)
toc
figure
plotconfusion(train_labels,train_YPred)
pause;

%% Classify Test Images and Compute Accuracy
% 
% tic
% test_YPred = classify(net2,test_set);
% test_accuracy = sum(test_YPred == test_labels)/numel(test_labels)
% toc
% figure
% plotconfusion(test_labels,test_YPred)

%% Live image capture demo
%https://www.mathworks.com/help/nnet/examples/visualize-activations-of-a-convolutional-neural-network.html

close all 

cam = webcam(1);
preview(cam);
pause;
img = snapshot(cam);
%imwrite(I,'Image06.jpg');
clear('cam')

%img = imread('image14.jpg');

img = rgb2gray(img);
 
figure
imshow(img,[])

pause;
%img = imgaussfilt(img,1);
BW = edge(img, 'canny', 0.5);
%BW = imcomplement(BW);
BW=bwareaopen(BW, 20);

se = strel('disk',10);
closeBW = imclose(BW,se);

img1 = double(img) .* double(closeBW);

stats=regionprops(closeBW ,'Area','BoundingBox','Centroid','Eccentricity','FilledArea','ConvexArea',...
    'ConvexImage');
[row, col]=size(stats);
if row~=1
    %disp('Take better image');
    f = msgbox(['The number is:',char(Ytest)]);
    
else
    img2 = imcrop(img1,[stats.BoundingBox(1)-100,stats.BoundingBox(2)-100, stats.BoundingBox(3)+200, stats.BoundingBox(4)+200]);
    img3 = imresize(img2,[28, 28]);
    img3 = adapthisteq(img3);
    figure
    imshow(img3,[])
    pause;
    % Test live image through trained Netwwork
    test(:,:,1,1) = img3;
    Ytest = classify(net,test);
    %disp(['The number is:',char(Ytest)])
    f = msgbox(['The number is:',char(Ytest)]);
end

%% Visualize image parameters
%imshow(img3,[])
% imgSize = size(img3);
% imgSize = imgSize(1:2);

%% Visualize Activations of a Convolutional Neural Network 
% 
% act1 = activations(net,img3,'conv_1','OutputAs','channels');
% sz = size(act1);
% act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
% figure
% montage(imresize(mat2gray(act1(:,:,:,[1:10])),imgSize))
% title(['Layer ','conv_1',' Activation image'])
% 
% % 
% % figure
% % for i = 1:10
% %     subplot(3,4,i)
% %     imshow(mat2gray(act1(:,:,:,i)),[])
% % end
% 
% act2 = activations(net,img3,'conv_2','OutputAs','channels');
% sz = size(act2);
% act2 = reshape(act2,[sz(1) sz(2) 1 sz(3)]);
% figure
% montage(imresize(mat2gray(act2(:,:,:,[1:10])),imgSize)) 
% title(['Layer ','conv_2',' Activation image'])
% % figure
% % for i = 1:10
% %     subplot(3,4,i)
% %     imshow(mat2gray(act2(:,:,:,i)),[])
% % end
% 
% act3 = activations(net,img3,'conv_3','OutputAs','channels');
% 
% sz = size(act3);
% act3 = reshape(act3,[sz(1) sz(2) 1 sz(3)]);
% figure
% montage(imresize(mat2gray(act3(:,:,:,[1:10])),imgSize))
% title(['Layer ','conv_3',' Activation image'])
% 
% % figure
% % for i = 1:10
% %     subplot(3,4,i)
% %     imshow(mat2gray(act3(:,:,:,i)),[])
% % end



%% comparing image and channel data

% act1ch5 = act1(:,:,:,1);
% act1ch5 = mat2gray(act1ch5);
% act1ch5 = imresize(act1ch5,imgSize);
% imshowpair(img3,act1ch5,'montage')

% %% Most strong activation conv 01
% 
% [~,maxValueIndex] = max(max(max(act1)));
% act1chMax = act1(:,:,:,maxValueIndex);
% act1chMax = mat2gray(act1chMax);
% act1chMax = imresize(act1chMax,imgSize);
% figure
% imshowpair(img3,act1chMax,'montage')
% 
% [~,maxValueIndex] = max(max(max(act2)));
% act2chMax = act1(:,:,:,maxValueIndex);
% act2chMax = mat2gray(act2chMax);
% act2chMax = imresize(act1chMax,imgSize);
% figure
% imshowpair(img3,act2chMax,'montage')
% 
% [~,maxValueIndex] = max(max(max(act3)));
% act3chMax = act1(:,:,:,maxValueIndex);
% act3chMax = mat2gray(act3chMax);
% act3chMax = imresize(act3chMax,imgSize);
% figure
% imshowpair(img3,act3chMax,'montage')


%% Activations after Relu

% act1relu = activations(net,img3,'relu_1','OutputAs','channels');
% sz = size(act1relu);
% act1relu = reshape(act1relu,[sz(1) sz(2) 1 sz(3)]);
% 
% 
% figure
% montage(imresize(mat2gray(act1relu(:,:,:,[1:10])),imgSize))
% title(['Layer ','relu_1',' Activation image'])
% 
% act2relu = activations(net,img3,'relu_2','OutputAs','channels');
% sz = size(act2relu);
% act2relu = reshape(act2relu,[sz(1) sz(2) 1 sz(3)]);
% 
% figure
% montage(imresize(mat2gray(act2relu(:,:,:,[1:10])),imgSize))
% title(['Layer ','relu_2',' Activation image'])
% 
% act3relu = activations(net,img3,'relu_3','OutputAs','channels');
% sz = size(act3relu);
% act3relu = reshape(act3relu,[sz(1) sz(2) 1 sz(3)]);
% 
% figure
% montage(imresize(mat2gray(act3relu(:,:,:,[1:10])),imgSize))
% title(['Layer ','relu_3',' Activation image'])

%% Activation of Fully connected network
% 
% actFC = activations(net,img3,'fc','OutputAs','channels');
% 
% sz = size(actFC);
% actFC = reshape(actFC,[sz(1) sz(2) 1 sz(3)]);
% 
% figure
% montage(imresize(mat2gray(actFC(:,:,:,[1:10])),imgSize))



