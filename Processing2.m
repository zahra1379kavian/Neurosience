clc
close all
clear all
%% read eeg data after preprocessing and plot data per each channel
clc; clear all;
T = readtable('text4.csv', 'Format', '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');
eegData4 = table2array(T);
time4 = eegData4(:,1);
T = readtable('text3.csv', 'Format', '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');
eegData3 = table2array(T);
time3 = eegData3(:,1);
T = readtable('text1.csv', 'Format', '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');
eegData1 = table2array(T);
time1 = eegData1(:,1);

eegData = [eegData4; eegData3; eegData1];
time = [time4; time3; time1];
epochShift = [0 length(eegData4(:,1)) length(eegData3(:,1))];
%% read tdms data
num = 150;num1 = 100;
tdmsData = zeros(1, num*2);
filename = 'alishahbazi1.tdms';
my_tdms_struct = TDMS_getStruct(filename);
tdms1 = my_tdms_struct.EEG_Raw.Trig.data;
s = find(diff(tdms1));s(s == 195978) = [];s(s == 195979) = [];
tdmsData(1, 2*num1+1: 3*num1) = s;
filename = 'alishahbazi3.tdms';
my_tdms_struct = TDMS_getStruct(filename);
tdms3 = my_tdms_struct.EEG_Raw.Trig.data;
s = find(diff(tdms3));
s(s == 74187) = [];s(s == 74186) = [];s(s == 1) = [];
tdmsData(1, num1+1: 2*num1) = s;
filename = 'alishahbazi4.tdms';
my_tdms_struct = TDMS_getStruct(filename);
tdms4 = my_tdms_struct.EEG_Raw.Trig.data;
s(s == 281653) = [];s(s == 281654) = [];s(s == 181226) = [];
s(s == 181227) = [];s(s == 1) = [];
tdmsData(1, 1: num1) = s;
%% read responseKeys
nans = NaN(50,2);
T = load('responseKey_block4.mat','-mat');
responseKeys4 = T.responseKey;

T = load('responseKey_block3.mat','-mat');
responseKeys3 = [T.responseKey, nans];

T = load('responseKey_block1.mat','-mat');
responseKeys1 = [T.responseKey, nans];

responseKeys = [responseKeys4; responseKeys3; responseKeys1];
responseKeys(responseKeys==0) = NaN;
%% read responseTimes
T = load('responseTime_block4.mat','-mat');
T = T.responseTime;
T(cellfun(@isempty,T)) = {NaN};
responseTimes4 = cell2mat(T);

T = load('responseTime_block3.mat','-mat');
T = T.responseTime;
T(cellfun(@isempty,T)) = {NaN};
responseTimes3 = [cell2mat(T), nans];

T = load('responseTime_block1.mat','-mat');
T = T.responseTime;
T(cellfun(@isempty,T)) = {NaN};
responseTimes1 = [cell2mat(T), nans];

responseTimes = [responseTimes4; responseTimes3; responseTimes1];

%% count number of NaNs
numberOfNaNs = 0;
for j = 1 : num
    for k = 1 : 12
        if (isnan(responseTimes(j, k)))
            numberOfNaNs = numberOfNaNs + 1;
        end
    end
end
%% count number of rights and lefts
rights = 0;
lefts = 0;
for i = 1:num
    for j = 1 : 12
        if(responseKeys(i, j) == 39)
            rights = rights + 1;
        elseif(responseKeys(i, j) == 37)
            lefts = lefts + 1;
        end
    end
end
%% create epoches based on responseTimes
% each epoch contains the time between one second before of response time until one second after of response time
% f = 250, number of responses = 50*12 - numberOfNaNs
clc
time_length = 2;
epoches = zeros(num*12, 250*time_length+1 , 30);
data = zeros(30, 250*time_length+1, num*12);
right_data = zeros(30, 250*time_length+1, rights);
left_data = zeros(30, 250*time_length+1, lefts);

tdms = tdmsData;


for k = 1: 30
    sr = 1; sl = 1;
    for i = 1 : num
        for j = 1 : 12
            if (~isnan(responseTimes(i, j)))
                epoches((i - 1)*12 + j, 1:501, k) = eegData((ceil(responseTimes(i, j)*250) - 250)+ tdms(2*i)+epochShift(ceil(i/50)) : (ceil(responseTimes(i, j)*250) + 250) + tdms(2*i) + epochShift(ceil(i/50)), k + 1);
                %data(k,1:250*time_length+1,(i - 1)*12 + j) = eegData((ceil(responseTimes(i, j))*250 - 249)+ tdms(2*i)+epochShift(ceil(num/50)) : (ceil(responseTimes(i, j))*250 + 250) + tdms(2*i) + epochShift(ceil(num/50)), k + 1);
				%epoches((i - 1)*12 + j, 501, k) = responseKeys(i, j);
				
				if responseKeys(i,j) == 39
                right_data(k,1:250*time_length+1,sr) =  eegData((ceil(responseTimes(i, j)*250) - 250)+ tdms(2*i)+epochShift(ceil(i/50)) : (ceil(responseTimes(i, j)*250) + 250) + tdms(2*i) + epochShift(ceil(i/50)), k + 1);  
                sr = sr+1;
                end
                if responseKeys(i,j) == 37
                left_data(k,1:250*time_length+1,sl) =  eegData((ceil(responseTimes(i, j)*250) - 250)+ tdms(2*i)+epochShift(ceil(i/50)) : (ceil(responseTimes(i, j)*250) + 250) + tdms(2*i) + epochShift(ceil(i/50)), k + 1);
                sl = sl+1;
                end
            else
                epoches((i - 1)*12 + j, 1:501,k) = NaN;
            end
        end
    end
end
%%
CH_name = ["Fp1","Fp2","F7","F3","Fz","F4","F8","FT7","FC3","FCz","FC4","FT8","T7","C3","Cz","C4","T8","TP7",...
    "CP3","CPz","CP4","TP8","P7","P3","Pz","P4","P8","O1","Oz","O2"];
%% Remove Empty Trial
for i=1:30
a = epoches(:,:,i);
a(any(isnan(a), 2), :) = [];
X(:,:,i) = a;
end
%% Average Evoked Potential
clc
close all
for k = 1:30
m = max(X(:,:,k),[],2);
s = 1;
for i = 1:size(X,1)
 if max(X(i,:,k)) >= 0 & max(X(i,:,k)) <= m+200
     m1(s,:) =  X(i,:,k);
     s = s+1;
  end
% if m(i)<10000
%    m1(s,:) =  X(i,:,k);
%    s = s+1; 
% end
end

figure
for j = 1:size(m1,1)
plot(m1(j,:),':')
hold on
end
plot(mean(m1,1),'black')
hold on
plot(250.*ones(size(-300:300)),-300:300,'r','LineWidth',2)
xlim([0 500])
title(["Channel "+CH_name(k)])
end
%
%%
close all
X1 = X;
X1(848,:,:)=[];
X1(491,:,:)=[];
X1(85,:,:)=[];
X = X1;
Average_trial = squeeze(nanmean(X,1));

for k = 1:30
figure
for i = 1:size(X,1)
plot(X(i,:,k),':')
hold on
end
plot(Average_trial(:,k),'black')
hold on
plot(250.*ones(size(-300:300)),-300:300,'r','LineWidth',2)
xlim([0 500])
title(["Channel "+CH_name(k)])
end
%% Spectogram
close all
noverlap = 63;
nfft = 512;
fs = 250;
X1 = X;
X1(848,:,:)=[];
X1(491,:,:)=[];
X1(85,:,:)=[];
for i = 1:30
figure
signal = mean(squeeze(X1(i,:,:)),2);
spectrogram(signal,	gausswin(64),noverlap,nfft,fs,'yaxis')
%title('Gausswin Window \\ Overlap = 0')
colormap('jet')
title(["Channel "+CH_name(i)])
end
%% graph theory
clc
s = 1;
for i = 1:10:size(X,1)
   data(:,:,s) = reshape(X(i,:,:),30,501); 
   s = s+1;
end

s = 1;
for i = 1:10:size(right_data,3)
   data_r(:,:,s) = reshape(right_data(:,:,i),30,501); 
   s = s+1;
end

s = 1;
for i = 1:10:size(left_data,3)
   data_l(:,:,s) = reshape(left_data(:,:,i),30,501); 
   s = s+1;
end
%% Seprate Frequency Band
filtSpec.order = 50;
filtSpec.range = [8 12]; %alpha range
plv_alpha = PLV_func(data, filtSpec);
filtSpec.range = [12 30]; %beta range
plv_beta = PLV_func(data, filtSpec);
filtSpec.range = [30 50]; %gamma range
plv_gamma = PLV_func(data, filtSpec);
%%
filtSpec.order = 50;
filtSpec.range = [8 12]; %alpha range
plv_alpha_r = PLV_func(data_r, filtSpec);
filtSpec.range = [12 30]; %beta range
plv_beta_r = PLV_func(data_r, filtSpec);
filtSpec.range = [30 50]; %gamma range
plv_gamma_r = PLV_func(data_r, filtSpec);
%%
filtSpec.order = 50;
filtSpec.range = [8 12]; %alpha range
plv_alpha_l = PLV_func(data_l, filtSpec);
filtSpec.range = [12 30]; %beta range
plv_beta_l = PLV_func(data_l, filtSpec);
filtSpec.range = [30 50]; %gamma range
plv_gamma_l = PLV_func(data_l, filtSpec);
%% PLV Adjency Matrix 
figure
subplot(1,3,1)
colormap('jet')
imagesc(squeeze(mean(plv_alpha,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Alpha Band')

subplot(1,3,2)
colormap('jet')
imagesc(squeeze(mean(plv_beta,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Beta Band')

subplot(1,3,3)
colormap('jet')
imagesc(squeeze(mean(plv_gamma,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Gamma Band')
colorbar
%%
figure
subplot(1,3,1)
colormap('jet')
imagesc(squeeze(mean(plv_alpha_r,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Alpha Band')

subplot(1,3,2)
colormap('jet')
imagesc(squeeze(mean(plv_beta_r,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Beta Band')

subplot(1,3,3)
colormap('jet')
imagesc(squeeze(mean(plv_gamma_r,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Gamma Band')
 
%%
figure
subplot(1,3,1)
colormap('jet')
imagesc(squeeze(mean(plv_alpha_l,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Alpha Band')

subplot(1,3,2)
colormap('jet')
imagesc(squeeze(mean(plv_beta_l,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Beta Band')

subplot(1,3,3)
colormap('jet')
imagesc(squeeze(mean(plv_gamma_l,1)))
xlabel('Electrodes')
ylabel('Electrodes')
title('Gamma Band')

%%
theta = [-18,18,-54,-39,0,39,54,-72,-62,0,62,72,-90,-90,90,90,90,-108,-118,180,118,108,-126,-141,180,141,...
    126,-162,180,162];
rho = [0.51111,0.51111,0.51111,0.33333,0.25556,0.33333,0.51111,0.51111,0.27778,0.12778,0.27778,0.51111,...
    0.5111,0.25556,0,0.25556,0.5111,0.5111,0.27778,0.12778,0.27778,0.51111,0.51111,0.33333,0.25556,0.3333,...
    0.51111,0.51111,0.51111,0.51111];

[elecx,elecy] = pol2cart(pi/180.*theta+180,rho);
%% PLV Graph
clc
mean_PLV = squeeze(mean(plv_alpha_l,1));
threshold1 = median(mean_PLV,'all');
s = 1;
s1 = 1;

figure
scatter(elecx,elecy,100,[206/255 206/255 206/255],'filled')
hold on

for i = 1:size(mean_PLV,1)
    for j = i:size(mean_PLV,2)
        if mean_PLV(i,j)>=threshold1 & mean_PLV(i,j)~=1
            line([elecx(i),elecx(j)],[elecy(i),elecy(j)],'linewidth',mean_PLV(i,j)+2,'Color','r');
            hold on
        end
    end
end
title('EEG graph base PLV, alpha band')
%%
clc
mean_PLV = squeeze(mean(plv_beta,1));
threshold1 = median(mean_PLV,'all');
s = 1;
s1 = 1;

figure
scatter(elecx,elecy,100,[0 0 0],'filled')
hold on

for i = 1:size(mean_PLV,1)
    for j = i:size(mean_PLV,2)
        if mean_PLV(i,j)>=threshold1 & mean_PLV(i,j)~=1
            line([elecx(i),elecx(j)],[elecy(i),elecy(j)],'linewidth',mean_PLV(i,j)+2,'Color','g');
            hold on
        end
    end
end

%% clustering
clc

plv_matrix_alpha = squeeze(mean(plv_alpha_r,1));
[acc_alpha, c_alpha_r] = weighted_avgClusteringCoefficient(plv_matrix_alpha);
plv_matrix_beta = squeeze(mean(plv_beta_r,1));
[acc_beta, c_beta_r] = weighted_avgClusteringCoefficient(plv_matrix_beta);
plv_matrix_gamma = squeeze(mean(plv_gamma_r,1));
[acc_gamma, c_gamma_r] = weighted_avgClusteringCoefficient(plv_matrix_gamma);
cc = [c_alpha_r,c_beta_r,c_gamma_r];
figure
boxplot(cc,'Labels',{'Alpha Band','Beta Band','Gamma Band'})
title('Clustering Coefficent Right Button')


plv_matrix_alpha = squeeze(mean(plv_alpha_l,1));
[acc_alpha, c_alpha_l] = weighted_avgClusteringCoefficient(plv_matrix_alpha);
plv_matrix_beta = squeeze(mean(plv_beta_l,1));
[acc_beta, c_beta_l] = weighted_avgClusteringCoefficient(plv_matrix_beta);
plv_matrix_gamma = squeeze(mean(plv_gamma_l,1));
[acc_gamma, c_gamma_l] = weighted_avgClusteringCoefficient(plv_matrix_gamma);
cc = [c_alpha_l,c_beta_l,c_gamma_l];
figure
boxplot(cc,'Labels',{'Alpha Band','Beta Band','Gamma Band'})
title('Clustering Coefficent Left Button')
%% global effeicency
plv_matrix_alpha = squeeze(mean(plv_alpha_r,1));
[WeightedEG_alpha, Wn_alpha] = global_efficency(plv_matrix_alpha);
plv_matrix_beta = squeeze(mean(plv_beta_r,1));
[WeightedEG_beta, Wn_beta] = global_efficency(plv_matrix_beta);
plv_matrix_gamma = squeeze(mean(plv_gamma_r,1));
[WeightedEG_gamma, Wn_gamma] = global_efficency(plv_matrix_gamma);

ge = [Wn_alpha,Wn_beta,Wn_gamma];
figure
boxplot(ge,'Labels',{'Alpha Band','Beta Band','Gamma Band'})
title('Global Efficency Right Button')

plv_matrix_alpha = squeeze(mean(plv_alpha_l,1));
[WeightedEG_alpha, Wn_alpha] = global_efficency(plv_matrix_alpha);
plv_matrix_beta = squeeze(mean(plv_beta_l,1));
[WeightedEG_beta, Wn_beta] = global_efficency(plv_matrix_beta);
plv_matrix_gamma = squeeze(mean(plv_gamma_l,1));
[WeightedEG_gamma, Wn_gamma] = global_efficency(plv_matrix_gamma);

ge = [Wn_alpha,Wn_beta,Wn_gamma];
figure
boxplot(ge,'Labels',{'Alpha Band','Beta Band','Gamma Band'})
title('lobal Efficency Left Button')
%% centrality
plv_matrix_alpha = squeeze(mean(plv_alpha_r,1));
binary_matrix_alpha = plv_matrix_alpha;
binary_matrix_alpha(binary_matrix_alpha>=mean(plv_matrix_alpha,'all')) = 1;
binary_matrix_alpha(binary_matrix_alpha<mean(plv_matrix_alpha,'all')) = 0;
G = graph(binary_matrix_alpha);
n = numnodes(G);
Centrality_degree_alpha = centrality(G,'degree')/((n-2)*(n-1)/2);
Centrality_betweenness_alpha = centrality(G,'betweenness')/((n-2)*(n-1)/2);

plv_matrix_beta = squeeze(mean(plv_beta_r,1));
binary_matrix_beta = plv_matrix_beta;
binary_matrix_beta(binary_matrix_beta>=mean(plv_matrix_beta,'all')) = 1;
binary_matrix_beta(binary_matrix_beta<mean(plv_matrix_beta,'all')) = 0;
G = graph(binary_matrix_beta);
n = numnodes(G);
Centrality_degree_beta = centrality(G,'degree')/((n-2)*(n-1)/2);
Centrality_betweenness_beta = centrality(G,'betweenness')/((n-2)*(n-1)/2);

plv_matrix_gamma = squeeze(mean(plv_gamma_r,1));
binary_matrix_gamma = plv_matrix_gamma;
binary_matrix_gamma(binary_matrix_gamma>=mean(plv_matrix_gamma,'all')) = 1;
binary_matrix_gamma(binary_matrix_gamma<mean(plv_matrix_gamma,'all')) = 0;
G = graph(binary_matrix_gamma);
n = numnodes(G);
Centrality_degree_gamma = centrality(G,'degree')/((n-2)*(n-1)/2);
Centrality_betweenness_gamma = centrality(G,'betweenness')/((n-2)*(n-1)/2);

%%
clc
ch_list = {'C4', 'CP4', 'F4', 'FC4', 'Cz','FP2', 'Fz', 'FCz', 'O2', 'TP8', 'P8', ...
     'FT8', 'T8', 'P4','F8', 'P3', 'CP3', 'C3', 'FC3', 'F3','FP1','Pz',...
     'CPz','Oz','O1','P7', 'TP7', 'T7', 'FT7', 'F7'};

plot_topography(ch_list, Centrality_degree_alpha, 0, '10-20', 0, 1, 1000, "Degree Centrality right button alpha band")
colormap jet
plot_topography(ch_list, Centrality_betweenness_alpha, 0, '10-20', 0, 1, 1000, "Betweenness Centrality right button alpha band")
colormap jet

plot_topography(ch_list, Centrality_degree_beta, 0, '10-20', 0, 1, 1000, "Degree Centrality right button beta band")
colormap jet
plot_topography(ch_list, Centrality_betweenness_beta, 0, '10-20', 0, 1, 1000, "Betweenness Centrality right button beta band")
colormap jet

plot_topography(ch_list, Centrality_degree_gamma, 0, '10-20', 0, 1, 1000, "Degree Centrality right button gamma band")
colormap jet
plot_topography(ch_list, Centrality_betweenness_gamma, 0, '10-20', 0, 1, 1000, "Betweenness Centrality right button gamma band")
colormap jet
%%
plv_matrix_alpha = squeeze(mean(plv_alpha_l,1));
binary_matrix_alpha = plv_matrix_alpha;
binary_matrix_alpha(binary_matrix_alpha>=mean(plv_matrix_alpha,'all')) = 1;
binary_matrix_alpha(binary_matrix_alpha<mean(plv_matrix_alpha,'all')) = 0;
G = graph(binary_matrix_alpha);
n = numnodes(G);
Centrality_degree_alpha = centrality(G,'degree')/((n-2)*(n-1)/2);
Centrality_betweenness_alpha = centrality(G,'betweenness')/((n-2)*(n-1)/2);

plv_matrix_beta = squeeze(mean(plv_beta_l,1));
binary_matrix_beta = plv_matrix_beta;
binary_matrix_beta(binary_matrix_beta>=mean(plv_matrix_beta,'all')) = 1;
binary_matrix_beta(binary_matrix_beta<mean(plv_matrix_beta,'all')) = 0;
G = graph(binary_matrix_beta);
n = numnodes(G);
Centrality_degree_beta = centrality(G,'degree')/((n-2)*(n-1)/2);
Centrality_betweenness_beta = centrality(G,'betweenness')/((n-2)*(n-1)/2);

plv_matrix_gamma = squeeze(mean(plv_gamma_l,1));
binary_matrix_gamma = plv_matrix_gamma;
binary_matrix_gamma(binary_matrix_gamma>=mean(plv_matrix_gamma,'all')) = 1;
binary_matrix_gamma(binary_matrix_gamma<mean(plv_matrix_gamma,'all')) = 0;
G = graph(binary_matrix_gamma);
n = numnodes(G);
Centrality_degree_gamma = centrality(G,'degree')/((n-2)*(n-1)/2);
Centrality_betweenness_gamma = centrality(G,'betweenness')/((n-2)*(n-1)/2);

%%
clc
close all
ch_list = {'C4', 'CP4', 'F4', 'FC4', 'Cz','FP2', 'Fz', 'FCz', 'O2', 'TP8', 'P8', ...
     'FT8', 'T8', 'P4','F8', 'P3', 'CP3', 'C3', 'FC3', 'F3','FP1','Pz',...
     'CPz','Oz','O1','P7', 'TP7', 'T7', 'FT7', 'F7'};

plot_topography(ch_list, Centrality_degree_alpha, 0, '10-20', 0, 1, 1000, "Degree Centrality left button alpha band")
colormap jet
plot_topography(ch_list, Centrality_betweenness_alpha, 0, '10-20', 0, 1, 1000, "Betweenness Centrality left button alpha band")
colormap jet

plot_topography(ch_list, Centrality_degree_beta, 0, '10-20', 0, 1, 1000, "Degree Centrality left button beta band")
colormap jet
plot_topography(ch_list, Centrality_betweenness_beta, 0, '10-20', 0, 1, 1000, "Betweenness Centrality left button beta band")
colormap jet

plot_topography(ch_list, Centrality_degree_gamma, 0, '10-20', 0, 1, 1000, "Degree Centrality left button gamma band")
colormap jet
plot_topography(ch_list, Centrality_betweenness_gamma, 0, '10-20', 0, 1, 1000, "Betweenness Centrality left button gamma band")
colormap jet
%%
s = 1;
for i = 1:40
for j = 2:10
   if isempty(responseTime{i, j})~= 1
       R1(s) = responseTime{i, j} - responseTime{i,j-1};
       s = s+1;
   end
end
end
figure
hist(R1)

%%
s = 1;
for i = 1:40
for j = 2:10
   if responseTimes(i, j) ~= 0
       R2(s) = responseTimes(i, j) - responseTimes(i,j-1);
       s = s+1;
   end
end
end


function [plv] = PLV_func(eegData, filtSpec)
numChannels = size(eegData, 1);
trial = size(eegData, 2);
numTrials = size(eegData, 3);

bpFilt = designfilt('bandpassfir','FilterOrder',filtSpec.order, ...
         'CutoffFrequency1',filtSpec.range(1),'CutoffFrequency2',filtSpec.range(2), ...
         'SampleRate',1000);
filteredData = zeros(size(eegData));
for i = 1:numChannels
filteredData(i,:,:) = filtfilt(bpFilt,squeeze(eegData(i,:,:)));
end
%filteredData = eegData;

angle_filteredData = zeros(size(eegData));
for channelCount = 1:numChannels
    angle_filteredData(channelCount, :, :) = angle(hilbert(filteredData(channelCount, :, :)));
end

plv = zeros(numTrials, numChannels, numChannels);

for channelCount = 1:numChannels
    for compareChannelCount = 1:numChannels
            if channelCount ~= compareChannelCount
            plv(:, channelCount, compareChannelCount) = abs(mean(squeeze(exp(1i*(angle_filteredData(channelCount,:,:) - angle_filteredData(compareChannelCount,:,:)))), 1));
            end
    end
end
end

function [acc, c] = weighted_avgClusteringCoefficient(graph)
deg = (size(graph,1)-1)*ones(size(graph,1),1); %Determine node degrees
cn = diag(graph.^(1/3)*triu(graph.^(1/3))*graph.^(1/3)); %Number of triangles for each node

%The local clustering coefficient of each node
c = zeros(size(deg));
c(deg > 1) = 2 * cn(deg > 1) ./ (deg(deg > 1).*(deg(deg > 1) - 1));

%Average clustering coefficient of the graph
acc = mean(c(deg > 1));
end

function [acc, c] = avgClusteringCoefficient(graph)
deg = sum(graph, 2); %Determine node degrees
cn = diag(graph*triu(graph)*graph); %Number of triangles for each node

%The local clustering coefficient of each node
c = zeros(size(deg));
c(deg > 1) = 2 * cn(deg > 1) ./ (deg(deg > 1).*(deg(deg > 1) - 1)); 

%Average clustering coefficient of the graph
acc = mean(c(deg > 1)); 
end

function [WeightedEG, Wn] = global_efficency(plv_matrix)
n = 30;
sp_plv = sparse(plv_matrix);
G = digraph(plv_matrix);
Gw = G.Edges.Weight;
Dtotal = graphallshortestpaths(sp_plv,'directed',false,'Weights',Gw);
Wn = ((sum(1./(Dtotal+eye(n))))./(n*(n-1)))';
WeightedEG = (sum(sum(1./(Dtotal+eye(n)))) - n)/(n*(n-1));
end

function h = plot_topography(ch_list, values, make_contour, system, plot_channels, plot_clabels, INTERP_POINTS, name)
figure;
    % Error detection
    if nargin < 2, error('[plot_topography] Not enough parameters.');
    else
        if ~iscell(ch_list) && ~ischar(ch_list)
            error('[plot_topography] ch_list must be "all" or a cell array.');
        end
        if ~isnumeric(values)
            error('[plot_topography] values must be a numeric vector.');
        end
    end
    if nargin < 3, make_contour = false;
    else
        if make_contour~=1 && make_contour~=0
            error('[plot_topography] make_contour must be a boolean (true or false).');
        end
    end
    if nargin < 4, system = '10-20';
    else
        if ~ischar(system) && ~istable(system)
            error('[plot_topography] system must be a string or a table.');
        end
    end
    if nargin < 5, plot_channels = true;
    else
        if plot_channels~=1 && plot_channels~=0
            error('[plot_topography] plot_channels must be a boolean (true or false).');
        end
    end
    if nargin < 5, plot_clabels = false;
    else
        if plot_clabels~=1 && plot_clabels~=0
            error('[plot_topography] plot_clabels must be a boolean (true or false).');
        end
    end
    if nargin < 6, INTERP_POINTS = 1000;
    else
        if ~isnumeric(INTERP_POINTS)
            error('[plot_topography] N must be an integer.');
        else
            if mod(INTERP_POINTS,1) ~= 0
                error('[plot_topography] N must be an integer.');
            end
        end
    end
    
    % Loading electrode locations
    if ischar(system)
        switch system
            case '10-20'
                % 10-20 system
                load('Standard_10-20_81ch.mat', 'locations');
            case '10-10'
                % 10-10 system
                load('Standard_10-10_47ch.mat', 'locations');
            case 'yokogawa'
                % Yokogawa MEG system
                load('MEG_Yokogawa-440ag.mat', 'locations');
            otherwise
                % Custom path
                load(system, 'locations');
        end
    else
        % Custom table
        locations = system;
    end
    
    % Finding the desired electrodes
    ch_list = upper(ch_list);
    if ~iscell(ch_list)
        if strcmp(ch_list,'all')
            idx = 1:length(locations.labels);
            if length(values) ~= length(idx)
                error('[plot_topography] There must be a value for each of the %i channels.', length(idx));
            end
        else, error('[plot_topography] ch_list must be "all" or a cell array.');
        end
    else
        if length(values) ~= length(ch_list)
            error('[plot_topography] values must have the same length as ch_list.');
        end
        idx = NaN(length(ch_list),1);
        for ch = 1:length(ch_list)
            if isempty(find(strcmp(locations.labels,ch_list{ch})))
                warning('[plot_topography] Cannot find the %s electrode.',ch_list{ch});
                ch_list{ch} = [];
                values(ch)  = [];
                idx(ch)     = [];
            else
                idx(ch) = find(strcmp(locations.labels,ch_list{ch}));
            end
        end
    end
    values = values(:);
    
    % Global parameters
    %   Note: Head radius should be set as 0.6388 if the 10-20 system is used.
    %   This number was calculated taking into account that the distance from Fpz
    %   to Oz is d=2*0.511. Thus, if the circle head must cross the nasion and
    %   the inion, it should be set at 5d/8 = 0.6388.
    %   Note2: When the number of interpolation points rises, the plots become
    %   smoother and more accurate, however, computational time also rises.
    HEAD_RADIUS     = 5*2*0.511/8;  % 1/2  of the nasion-inion distance
    HEAD_EXTRA      = 1*2*0.511/8;  % 1/10 of the nasion-inion distance
    k = 4;                          % Number of nearest neighbors for interpolation
    
    % Interpolating input data
        % Creating the rectangle grid (-1,1)
        [ch_x, ch_y] = pol2cart((pi/180).*((-1).*locations.theta(idx)+90), ...
                                locations.radius(idx));     % X, Y channel coords
        % Points out of the head to reach more natural interpolation
        r_ext_points = 1.2;
        [add_x, add_y] = pol2cart(0:pi/4:7*pi/4,r_ext_points*ones(1,8));
        linear_grid = linspace(-r_ext_points,r_ext_points,INTERP_POINTS);         % Linear grid (-1,1)
        [interp_x, interp_y] = meshgrid(linear_grid, linear_grid);
        
        % Interpolate and create the mask
        outer_rho = max(locations.radius(idx));
        if outer_rho > HEAD_RADIUS, mask_radius = outer_rho + HEAD_EXTRA;
        else,                       mask_radius = HEAD_RADIUS;
        end
        mask = (sqrt(interp_x.^2 + interp_y.^2) <= mask_radius); 
        add_values = compute_nearest_values([add_x(:), add_y(:)], [ch_x(:), ch_y(:)], values(:), k);
        interp_z = griddata([ch_x(:); add_x(:)], [ch_y(:); add_y(:)], [values; add_values(:)], interp_x, interp_y, 'natural');
        interp_z(mask == 0) = NaN;
        % Plotting the final interpolation
        pcolor(interp_x, interp_y, interp_z);
        shading interp;
        hold on;
        
        % Contour
        if make_contour
            [~, hfigc] = contour(interp_x, interp_y, interp_z); 
            set(hfigc, 'LineWidth',0.75, 'Color', [0.2 0.2 0.2]); 
            hold on;
        end
    % Plotting the head limits as a circle         
    head_rho    = HEAD_RADIUS;                      % Head radius
    if strcmp(system,'yokogawa'), head_rho = 0.45; end
    head_theta  = linspace(0,2*pi,INTERP_POINTS);   % From 0 to 360Âº
    head_x      = head_rho.*cos(head_theta);        % Cartesian X of the head
    head_y      = head_rho.*sin(head_theta);        % Cartesian Y of the head
    plot(head_x, head_y, 'Color', 'k', 'LineWidth',4);
    hold on;
    % Plotting the nose
    nt = 0.15;      % Half-nose width (in percentage of pi/2)
    nr = 0.22;      % Nose length (in radius units)
    nose_rho   = [head_rho, head_rho+head_rho*nr, head_rho];
    nose_theta = [(pi/2)+(nt*pi/2), pi/2, (pi/2)-(nt*pi/2)];
    nose_x     = nose_rho.*cos(nose_theta);
    nose_y     = nose_rho.*sin(nose_theta);
    plot(nose_x, nose_y, 'Color', 'k', 'LineWidth',4);
    hold on;
    % Plotting the ears as ellipses
    ellipse_a = 0.08;                               % Horizontal exentricity
    ellipse_b = 0.16;                               % Vertical exentricity
    ear_angle = 0.9*pi/8;                           % Mask angle
    offset    = 0.05*HEAD_RADIUS;                   % Ear offset
    ear_rho   = @(ear_theta) 1./(sqrt(((cos(ear_theta).^2)./(ellipse_a^2)) ...
        +((sin(ear_theta).^2)./(ellipse_b^2))));    % Ellipse formula in polar coords
    ear_theta_right = linspace(-pi/2-ear_angle,pi/2+ear_angle,INTERP_POINTS);
    ear_theta_left  = linspace(pi/2-ear_angle,3*pi/2+ear_angle,INTERP_POINTS);
    ear_x_right = ear_rho(ear_theta_right).*cos(ear_theta_right);          
    ear_y_right = ear_rho(ear_theta_right).*sin(ear_theta_right); 
    ear_x_left  = ear_rho(ear_theta_left).*cos(ear_theta_left);         
    ear_y_left  = ear_rho(ear_theta_left).*sin(ear_theta_left); 
    plot(ear_x_right+head_rho+offset, ear_y_right, 'Color', 'k', 'LineWidth',4); hold on;
    plot(ear_x_left-head_rho-offset, ear_y_left, 'Color', 'k', 'LineWidth',4); hold on;
    % Plotting the electrodes
    % [ch_x, ch_y] = pol2cart((pi/180).*(locations.theta(idx)+90), locations.radius(idx));
    if plot_channels, he = scatter(ch_x, ch_y, 60,'k', 'LineWidth',1.5); end
    if plot_clabels, text(ch_x, ch_y, ch_list); end
    if strcmp(system,'yokogawa'), delete(he); plot(ch_x, ch_y, '.k'); end
    
    % Last considerations
    max_height = max([max(nose_y), mask_radius]);
    min_height = -mask_radius;
    max_width  = max([max(ear_x_right+head_rho+offset), mask_radius]);
    min_width  = -max_width;
    L = max([min_height, max_height, min_width, max_width]);
    xlim([-L, L]);
    ylim([-L, L]);  
    colorbar;   % Feel free to modify caxis after calling the function
    title(sprintf('Topographical map of %s power', name))
    axis square;
    axis off;
    hold off;
    h = gcf;
end
function add_val = compute_nearest_values(coor_add, coor_neigh, val_neigh, k)
    
    add_val = NaN(size(coor_add,1),1);
    L = length(add_val);
    
    for i = 1:L
        % Distances between the added electrode and the original ones
        target = repmat(coor_add(i,:),size(coor_neigh,1),1);
        d = sqrt(sum((target-coor_neigh).^2,2));
        
        % K-nearest neighbors
        [~, idx] = sort(d,'ascend');
        idx = idx(2:1+k);
        
        % Final value as the mean value of the k-nearest neighbors
        add_val(i) = mean(val_neigh(idx));
    end
    
end

