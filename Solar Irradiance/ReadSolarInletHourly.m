clear all
close all
clc

%%
filename1 = 'JAN 2021 Solar Irradiance.xlsx';
sheet1 = 1;
xlRange1 = 'D2:D3757'; 


%plot load data 
subsetA = xlsread(filename1,sheet1,xlRange1);
data1=abs(subsetA)
mean_subsetA=mean(subsetA);
max_subsetA=max(subsetA);
min_subsetA=min(subsetA);

%% convert to hourly

[m,n]=size(data1)
data_hourly=[];

%%

% N = m;
start_val = 1;
inc = 11;
stop_val = m
v = [start_val:inc:stop_val];


[m1,n1]=size(v)

%% 

counter=1;
for i=1:1:n1-1
    data_hourly(counter)=mean(data1(v(i):v(i+1)));
    counter=counter+1;
end

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

% [m2,n2]=size(dataB)
data_hourly2=[];

%%

% N = m;
start_val = 1;
inc = 11;
stop_val = m2
v = [start_val:inc:stop_val];

[m2,n2]=size(v)

%% 

counter=1;
for i=1:1:n2-1
    data_hourly(counter)=mean(data1(v(i):v(i+1)));
    counter=counter+1;
end

