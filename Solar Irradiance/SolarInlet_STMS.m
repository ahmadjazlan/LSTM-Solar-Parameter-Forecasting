clear all
close all
clc

%%

filename1 = 'JAN 2021 Solar Irradiance.xlsx';
sheet1 = 1;
xlRange1 = 'D2:D3757'; 


%plot load data 
subsetA = xlsread(filename1,sheet1,xlRange1);
data=abs(subsetA);
mean_subsetA=mean(subsetA);
max_subsetA=max(subsetA);
min_subsetA=min(subsetA);

%  figure
%  plot(data)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename2 = 'FEB 2021 Solar Irradiance.xlsx';
sheet2 = 1;
xlRange2 = 'D2:D3681';


%plot load data 
subsetB = xlsread(filename2,sheet2,xlRange2);
dataB=abs(subsetB);
mean_subsetB=mean(subsetB);
max_subsetB=max(subsetB);
min_subsetB=min(subsetB);
%  figure
%  plot(dataB)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")


%%

filename3 = 'MAR 2021 Solar Irradiance.xlsx';
sheet3 = 1;
xlRange3 = 'D2:D3836';


%plot load data 
subsetC = xlsread(filename3,sheet3,xlRange3);
dataC=abs(subsetC);
mean_subsetC=mean(subsetC);
max_subsetC=max(subsetC);
min_subsetC=min(subsetC);
%  figure
%  plot(dataC)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename4 = 'APR 2021 Solar Irradiance.xlsx';
sheet4 = 1;
xlRange4 = 'D2:D3880';


%plot load data 
subsetD = xlsread(filename4,sheet4,xlRange4);
dataD=abs(subsetD);
mean_subsetD=mean(subsetD);
max_subsetD=max(subsetD);
min_subsetD=min(subsetD);
%  figure
%  plot(dataD)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename5 = 'MAY 2021 Solar Irradiance.xlsx';
sheet5 = 1;
xlRange5 = 'D2:D4073';


%plot load data 
subsetE = xlsread(filename5,sheet5,xlRange5);
dataE=abs(subsetE);
mean_subsetE=mean(subsetE);
max_subsetE=max(subsetE);
min_subsetE=min(subsetE);
%  figure
%  plot(dataE)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename6 = 'JUN 2021 Solar Irradiance.xlsx';
sheet6 = 1;
xlRange6 = 'D2:D4073';


%plot load data 
subsetF = xlsread(filename6,sheet6,xlRange6);
dataF=abs(subsetF);
mean_subsetF=mean(subsetF);
max_subsetF=max(subsetF);
min_subsetF=min(subsetF);
%  figure
%  plot(dataF)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename7 = 'JUL 2021 Solar Irradiance.xlsx';
sheet7 = 1;
xlRange7 = 'D2:D4901';


%plot load data 
subsetG = xlsread(filename7,sheet7,xlRange7);
dataG=abs(subsetG);
mean_subsetG=mean(subsetG);
max_subsetG=max(subsetG);
min_subsetG=min(subsetG);

%  figure
%  plot(dataG)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")


%%

filename8 = 'AUG 2021 Solar Irradiance.xlsx';
sheet8 = 1;
xlRange8 = 'D2:D4092';


%plot load data 
subsetH = xlsread(filename8,sheet8,xlRange8);
dataH=abs(subsetH);
mean_subsetH=mean(subsetH);
max_subsetH=max(subsetH);
min_subsetH=min(subsetH);

%  figure
%  plot(dataH)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename9 = 'SEP 2021 Solar Irradiance.xlsx';
sheet9 = 1;
xlRange9 = 'D2:D3915';


%plot load data 
subsetI = xlsread(filename9,sheet9,xlRange9);
dataI=abs(subsetI);
mean_dataI=mean(dataI);
max_dataI=max(dataI);
min_dataI=min(dataI);
 
%  figure
%  plot(dataI)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

%%

filename10 = 'OCT 2021 Solar Irradiance.xlsx';
sheet10 = 1;
xlRange10 = 'D2:D4082';


%plot load data 
subsetJ = xlsread(filename10,sheet10,xlRange10);
dataJ=abs(subsetJ);
mean_dataJ=mean(dataJ);
max_dataJ=max(dataJ);
min_dataJ=min(dataJ);

%  figure
%  plot(dataJ)
%  xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")


%%

filename11 = 'NOV 2021 Solar Irradiance.xlsx';
sheet11 = 1;
xlRange11 = 'D2:D3954';


%plot load data 
subsetK = xlsread(filename11,sheet11,xlRange11);
dataK=abs(subsetK);

% figure
% plot(dataK)
% xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")

 %%
 
filename12 = 'DEC 2021 Solar Irradiance.xlsx';
sheet12 = 1;
xlRange12 = 'D2:D4025';


%plot load data 
subsetL = xlsread(filename12,sheet12,xlRange12);
dataL=abs(subsetL);

% figure
% plot(dataL)
% xlabel("Samples")
%  ylabel("Solar Irradiance (W/m^2)")
%  title("Solar Irradiance (W/m^2)")
%%

% for testing only (1 month)
% total_irradiance=[data(1:1000,:)];

% All Data
total_irradiance=[data; dataB; dataC; dataD; dataE; dataF; dataG; dataH; dataI; dataJ; dataK; dataL];

% total_HeatMeter1=[dataG; dataH; dataI; dataJ];

%% 90 % for training, 10 % for testing

numTimeStepsTrain = floor(0.9*numel(total_irradiance));

dataTrain = total_irradiance(1:numTimeStepsTrain+1);
dataTest = total_irradiance(numTimeStepsTrain+1:end);

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
YTest = dataTest(2:end);

%%
% 
% 
% %%
% 
% net = predictAndUpdateState(net,XTrain');
% [net,YPred] = predictAndUpdateState(net,YTrain(end));
% 
% numTimeStepsTest = numel(XTest);
% for i = 2:numTimeStepsTest
%     [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
% end
% 
% %% Unstandardize the predictions using the parameters calculated earlier.
% 
% YPred = sig*YPred + mu;
% 
% %% The training progress plot reports the root-mean-square error (RMSE) calculated from the standardized data. 
% %Calculate the RMSE from the unstandardized predictions.
% 
% %include RMSE formula in paper
% YTest = dataTest(2:end);
% rmse = sqrt(mean((YPred-YTest').^2))
% 
% %% Plot the training time series with the forecasted values.
% 
% figure
% plot(dataTrain(1:end-1))
% hold on
% 
% %%
% 
% % idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
% % plot(idx,[data(numTimeStepsTrain) YPred],'.-')
% % hold off
% % xlabel("Month")
% % ylabel("Cases")
% % title("Forecast")
% % %legend(["Observed" "Forecast"])
% 
% %% 
% 
% figure
% subplot(2,1,1)
% plot(YTest)
% hold on
% plot(YPred,'.-')
% hold off
% legend(["Observed" "Forecast"])
% ylabel("Cases")
% title("Forecast")
% 
% subplot(2,1,2)
% stem(YPred - YTest')
% xlabel("Month")
% ylabel("Error")
% title("RMSE = " + rmse)

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
ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
title("Collector Inlet Temperature",'fontweight','bold','FontSize', 15)

subplot(2,1,2)
stem(YPred - YTest')
xlabel("Samples",'fontweight','bold','FontSize', 15)
ylabel("Error",'fontweight','bold','FontSize', 15)
title("RMSE = " + rmse,'fontweight','bold','FontSize', 15)

% Here, the predictions are more accurate when updating the network state 
% with the observed values instead of the predicted values.