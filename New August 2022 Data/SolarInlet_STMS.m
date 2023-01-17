clear all
close all
clc

%%

filename1 = 'JAN 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet1 = 1;
xlRange1 = 'E2:E3757'; %Readings taken hourly
%xlRange1 = 'E2:E30'; %Readings taken hourly

%plot load data 
subsetA = xlsread(filename1,sheet1,xlRange1);
data=abs(subsetA);
mean_subsetA=mean(subsetA);
max_subsetA=max(subsetA);
min_subsetA=min(subsetA);

%  figure
%  plot(data)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Collector Inlet Temperature")

%%

filename2 = 'FEB 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet2 = 1;
xlRange2 = 'E2:E3681';


%plot load data 
subsetB = xlsread(filename2,sheet2,xlRange2);
dataB=abs(subsetB);
mean_subsetB=mean(subsetB);
max_subsetB=max(subsetB);
min_subsetB=min(subsetB);
%  figure
%  plot(dataB)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Collector Inlet Temperature")

%%

filename3 = 'MAR 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet3 = 1;
xlRange3 = 'E2:E3836';


%plot load data 
subsetC = xlsread(filename3,sheet3,xlRange3);
dataC=abs(subsetC);
mean_subsetC=mean(subsetC);
max_subsetC=max(subsetC);
min_subsetC=min(subsetC);
%  figure
%  plot(dataC)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Collector Inlet Temperature")

%%

filename4 = 'APR 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet4 = 1;
xlRange4 = 'E2:E3885';


%plot load data 
subsetD = xlsread(filename4,sheet4,xlRange4);
dataD=abs(subsetD);
mean_subsetD=mean(subsetD);
max_subsetD=max(subsetD);
min_subsetD=min(subsetD);
%  figure
%  plot(dataD)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Collector Inlet Temperature")

%%

filename5 = 'MAY 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet5 = 1;
xlRange5 = 'E2:E4073';


%plot load data 
subsetE = xlsread(filename5,sheet5,xlRange5);
dataE=abs(subsetE);
mean_subsetE=mean(subsetE);
max_subsetE=max(subsetE);
min_subsetE=min(subsetE);
%  figure
%  plot(dataE)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Collector Inlet Temperature")

%%

filename6 = 'JUN 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet6 = 1;
xlRange6 = 'E2:E3951';


%plot load data 
subsetF = xlsread(filename6,sheet6,xlRange6);
dataF=abs(subsetF);
mean_subsetF=mean(subsetF);
max_subsetF=max(subsetF);
min_subsetF=min(subsetF);
%  figure
%  plot(dataF)
%  xlabel("Samples")
%  ylabel("Temperature - Celsius")
%  title("Collector Inlet Temperature")

%%

filename7 = 'JUL 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet7 = 1;
xlRange7 = 'E2:E4091';


%plot load data 
subsetG = xlsread(filename7,sheet7,xlRange7);
dataG=abs(subsetG);
mean_subsetG=mean(subsetG);
max_subsetG=max(subsetG);
min_subsetG=min(subsetG);
% % figure
%  figure
%  plot(dataG,'LineWidth',1.5)
% grid on
% xlabel("Samples",'fontweight','bold','FontSize', 15)
% ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
% title("Collector Inlet Temperature - July 2021",'fontweight','bold','FontSize', 15)

%%

filename8 = 'AUG 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet8 = 1;
xlRange8 = 'E2:E4092';


%plot load data 
subsetH = xlsread(filename8,sheet8,xlRange8);
dataH=abs(subsetH);
mean_subsetH=mean(subsetH)
max_subsetH=max(subsetH)
min_subsetH=min(subsetH)

% figure
% plot(dataH,'LineWidth',1.5)
% grid on
% xlabel("Samples",'fontweight','bold','FontSize', 15)
% ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
% title("Collector Inlet Temperature - August 2021",'fontweight','bold','FontSize', 15)


%%

filename9 = 'SEP 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet9 = 1;
xlRange9 = 'E2:E3915';


%plot load data 
subsetI = xlsread(filename9,sheet9,xlRange9);
dataI=abs(subsetI);
mean_dataI=mean(dataI)
max_dataI=max(dataI)
min_dataI=min(dataI)
% figure
% plot(dataJ)
% xlabel("Samples")
% ylabel("Energy Per Hour kWh")
% title("Heat Meter 1 Readings")


%%

filename10 = 'OCT 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet10 = 1;
xlRange10 = 'E2:E4082';


%plot load data 
subsetK = xlsread(filename10,sheet10,xlRange10);
dataK=abs(subsetK);
% figure
% plot(dataK)
% xlabel("Samples")
% ylabel("Energy Per Hour kWh")
% title("Heat Meter 1 Readings")


%%

filename11 = 'NOV 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet11 = 1;
xlRange11 = 'E2:E3954';


%plot load data 
subsetL = xlsread(filename11,sheet11,xlRange11);
dataL=abs(subsetL);
% figure
% plot(dataK)
% xlabel("Samples")
% ylabel("Energy Per Hour kWh")
% title("Heat Meter 1 Readings")

%total_irradiance=[data; dataB; dataC; dataD; dataE; dataF; dataG; dataH; dataI; dataJ; dataK];

%total_HeatMeter1=[dataG; dataH; dataI; dataJ];
%%

filename12 = 'NOV 2021 Temperature Supply_Inlet_Outlet.xlsx';
sheet12 = 1;
xlRange12 = 'E2:E3954';


%plot load data 
subsetM = xlsread(filename12,sheet12,xlRange12);
dataM=abs(subsetM);
% figure
% plot(dataK)
% xlabel("Samples")
% ylabel("Energy Per Hour kWh")
% title("Heat Meter 1 Readings")

total_inlet=[data; dataB; dataC; dataD; dataE; dataF; dataG; dataH; dataI; dataJ; dataK; dataL; dataM];

%total_HeatMeter1=[dataG; dataH; dataI; dataJ];

%total_inlet=total_inlet(1:200)

%%

numTimeStepsTrain = floor(0.9*numel(total_inlet));

dataTrain = total_inlet(1:numTimeStepsTrain+1);
dataTest = total_inlet(numTimeStepsTrain+1:end);

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
ylabel("Temperature - Celsius",'fontweight','bold','FontSize', 15)
title("Collector Inlet Temperature",'fontweight','bold','FontSize', 15)

subplot(2,1,2)
stem(YPred - YTest')
xlabel("Samples",'fontweight','bold','FontSize', 15)
ylabel("Error",'fontweight','bold','FontSize', 15)
title("RMSE = " + rmse,'fontweight','bold','FontSize', 15)

% Here, the predictions are more accurate when updating the network state 
% with the observed values instead of the predicted values.