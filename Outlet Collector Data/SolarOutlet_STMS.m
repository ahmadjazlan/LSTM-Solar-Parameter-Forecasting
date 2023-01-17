clear all
close all
clc

%%

filename1 = 'STMS IOI January Outlet Temp.xlsx';
sheet1 = 1;
xlRange1 = 'B2:B8116'; %Readings taken hourly


%plot load data 
subsetA = xlsread(filename1,sheet1,xlRange1);
data=abs(subsetA);
mean_data=mean(data)
min_data=min(data)
max_data=max(data)
%  figure
%  plot(data)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%

filename2 = 'STMS IOI February Outlet Temp.xlsx';
sheet2 = 1;
xlRange2 = 'B2:B8034';

subsetB = xlsread(filename2,sheet2,xlRange2);

dataB=abs(subsetB);
mean_dataB=mean(dataB)
min_dataB=min(dataB)
max_dataB=max(dataB)
%  figure
%  plot(dataB)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%

filename3 = 'STMS IOI March Outlet Temp.xlsx';
sheet3 = 1;
xlRange3 = 'B2:B8411';

subsetC = xlsread(filename3,sheet3,xlRange3);

dataC=abs(subsetC);
mean_dataC=mean(dataC)
min_dataC=min(dataC)
max_dataC=max(dataC)
%  figure
%  plot(dataC)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%

filename4 = 'STMS IOI April Outlet Temp.xlsx';
sheet4 = 1;
xlRange4 = 'B2:B8411';

subsetD = xlsread(filename4,sheet4,xlRange4);

dataD=abs(subsetD);
mean_dataD=mean(dataD)
min_dataD=min(dataD)
max_dataD=max(dataD)
%  figure
%  plot(dataD)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%

filename5 = 'STMS IOI May Outlet Temp.xlsx';
sheet5 = 1;
xlRange5 = 'B2:B8908';

subsetE = xlsread(filename5,sheet5,xlRange5);

dataE=abs(subsetE);
mean_dataE=mean(dataE)
min_dataE=min(dataE)
max_dataE=max(dataE)
%  figure
%  plot(dataE)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")
%%

filename6 = 'STMS IOI June Outlet Temp.xlsx';
sheet6 = 1;
xlRange6 = 'B2:B8630';


%plot load data 
subsetF = xlsread(filename6,sheet6,xlRange6);
dataF=abs(subsetF);
mean_dataF=mean(dataF)
min_dataF=min(dataF)
max_dataF=max(dataF)
%  figure
%  plot(dataF)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%

filename7 = 'STMS IOI July Outlet Temp.xlsx';
sheet7 = 1;
xlRange7 = 'B2:B8924';


%plot load data 
subsetG = xlsread(filename7,sheet7,xlRange7);
dataG=abs(subsetG);
mean_dataG=mean(dataG)
min_dataG=min(dataG)
max_dataG=max(dataG)
% % figure
%  figure
%  plot(dataG,'LineWidth',1.5)
% grid on
% xlabel("Samples",'fontweight','bold','FontSize', 15)
% ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
% title("Solar Collector Outlet Temperature - July 2021",'fontweight','bold','FontSize', 15)

%%

filename8 = 'STMS IOI August Outlet Temp.xlsx';
sheet8 = 1;
xlRange8 = 'B2:B8925';


%plot load data 
subsetH = xlsread(filename8,sheet8,xlRange8);
dataH=abs(subsetH);
mean_dataH=mean(dataH)
min_dataH=min(dataH)
max_dataH=max(dataH)
% figure
% plot(dataH,'LineWidth',1.5)
% grid on
% xlabel("Samples",'fontweight','bold','FontSize', 15)
% ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
% title("Solar Collector Outlet Temperature - August 2021",'fontweight','bold','FontSize', 15)

%%
filename9 = 'STMS IOI September Outlet Temp.xlsx';
sheet9 = 1;
xlRange9 = 'B2:B8638';


%plot load data 
subsetI = xlsread(filename9,sheet9,xlRange9);
dataI=abs(subsetI);
mean_dataI=mean(dataI)
min_dataI=min(dataI)
max_dataI=max(dataI)
% figure
% plot(dataI,'LineWidth',1.5)
% grid on
% xlabel("Samples",'fontweight','bold','FontSize', 15)
% ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
% title("Solar Collector Outlet Temperature - September 2021",'fontweight','bold','FontSize', 15)
%%

filename10 = 'STMS IOI October Outlet Temp.xlsx';
sheet10 = 1;
xlRange10 = 'B2:B8857';


%plot load data 
subsetJ = xlsread(filename10,sheet10,xlRange10);
dataJ=abs(subsetJ);
mean_dataJ=mean(dataJ)
min_dataJ=min(dataJ)
max_dataJ=max(dataJ)
% figure
% plot(dataJ)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%

filename11 = 'STMS IOI November Outlet Temp.xlsx';
sheet11 = 1;
xlRange11 = 'B2:B8630';


%plot load data 
subsetK = xlsread(filename11,sheet11,xlRange11);
dataK=abs(subsetK);
% figure
% plot(dataK)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Solar Collector Outlet Temperature")

%%


%total_irradiance=[data; dataB; dataC; dataD; dataE; dataF; dataG; dataH; dataI; dataJ; dataK];

total_collector_output=[dataG; dataH; dataI; dataJ];

%%

numTimeStepsTrain = floor(0.9*numel(total_collector_output));

dataTrain = total_collector_output(1:numTimeStepsTrain+1);
dataTest = total_collector_output(numTimeStepsTrain+1:end);

%% Standardize Data

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

%% Prepare Predictors and Responses

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%% Define LSTM Network Architecture

%Create an LSTM regression network. Specify the LSTM layer to have 200 hidden units.
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%Specify the training options. 
%Set the solver to 'adam' and train for 250 epochs. 
%To prevent the gradients from exploding, set the gradient threshold to 1. 
%Specify the initial learn rate 0.005, and drop the learn rate after 125 epochs 
% by multiplying by a factor of 0.2.

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Train the LSTM network with the specified training options by using trainNetwork

net = trainNetwork(XTrain',YTrain',layers,options);

%% Forecast Future Time Steps

%To forecast the values of multiple time steps in the future, 
%use the predictAndUpdateState function to predict time steps one at a time and 
%update the network state at each prediction. 
%For each prediction, use the previous prediction as input to the function.

%Standardize the test data using the same parameters as the training data.

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);


%%

net = predictAndUpdateState(net,XTrain');
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%% Unstandardize the predictions using the parameters calculated earlier.

YPred = sig*YPred + mu;

%% The training progress plot reports the root-mean-square error (RMSE) calculated from the standardized data. 
%Calculate the RMSE from the unstandardized predictions.

%include RMSE formula in paper
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest').^2))

%% Plot the training time series with the forecasted values.

figure
plot(dataTrain(1:end-1))
hold on

%%

% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
% plot(idx,[data(numTimeStepsTrain) YPred],'.-')
% hold off
% xlabel("Month")
% ylabel("Cases")
% title("Forecast")
% %legend(["Observed" "Forecast"])

%% 

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest')
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)

%% Update Network State with Observed Values

% day ahead forecasting

net = resetState(net);
net = predictAndUpdateState(net,XTrain');

XTest=XTest';

%%
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

%%

YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest').^2))

%%

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Temperature (Celsius)",'fontweight','bold','FontSize', 15)
title("Collector Outlet Temperature",'fontweight','bold','FontSize', 15)

subplot(2,1,2)
stem(YPred - YTest')
xlabel("Samples",'fontweight','bold','FontSize', 15)
ylabel("Error",'fontweight','bold','FontSize', 15)
title("RMSE = " + rmse,'fontweight','bold','FontSize', 15)

% Here, the predictions are more accurate when updating the network state 
% with the observed values instead of the predicted values.