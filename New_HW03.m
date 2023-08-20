% Homeworke 03
clc
clear
close all

% Load data
load('S_monkey1.mat');
data= load('SCom_monkey1.mat');

% Initial parameters
num_neurons = size(S(1).trial(1).spikes,1);
num_stimulation = size(S,2);
num_trials = size(S(1).trial,2);
orientation= 0:30:330;

%% Plot PSTH

fig= figure;
for i=1:num_stimulation
    subplot(6,2,i)
    [~,I(i)]= max(max(S(i).mean_FRs,[],2));
    for j=1:num_neurons
        if j==I(i)
            plot(S(i).mean_FRs(j,:),'LineWidth',4,'Color','red'); hold on;
        end
        plot(S(i).mean_FRs(j,:),'LineWidth',0.3,'Color',[182 182 182]./255); hold on;
    end
    title("PSTH Condition "+i+", The Most Active Neuron "+I(i),'Interpreter','latex')
end

han= axes(fig,'visible','off');
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Spike/s','Interpreter','latex');
xlabel(han,'Time(s)','Interpreter','latex');


%% Tuning Curve
close all
% Average neuron response in trials
for i=1:num_stimulation
    for j=1:num_trials
        s(:,j)= sum(S(i).trial(j).counts,2);
    end
    mean_fire_trial(:,i)= mean(s,2);
end

mean_fire_stimul= mean(mean_fire_trial,2);

% Find the most active neuron on each condition

[B, neuron_indx]= sort(mean_fire_stimul,'descend');
fig=figure;

for i=1:12
    subplot(4,3,i)
    plot(orientation,mean_fire_trial(neuron_indx(i),:),'LineWidth',3);
    title("Tuning Curve Neuron "+neuron_indx(i)+" ,Monkey 1",'Interpreter','latex')
end
han=axes(fig,'visible','off');
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Mean Firing','Interpreter','latex');
xlabel(han,'Orientation','Interpreter','latex');


% Plot for each condition
cond= 0:30:330;
figure
for i=1:num_stimulation
    subplot(6,2,i)
    plot(orientation,mean_fire_trial(I(i),:),'LineWidth',1);
    title("Tuning Curve "+cond(i)+" Degree"+", Neuron "+neuron_indx(i)+" ,Monkey 1",'Interpreter','latex')
end
han=axes(fig,'visible','off');
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Mean Firing','Interpreter','latex');
xlabel(han,'Orientation','Interpreter','latex');

% Plot all neuron in all orientations
x= max(mean_fire_trial,[],1);
figure
imagesc(mean_fire_trial'./x','XData',1:80,'YData',0:30:330)
colormap summer
yticks(0:30:330)
xlabel('Neuron Number','Interpreter','latex')
ylabel('Orientation','Interpreter','latex')
title('Compare Neuron Activity in All Orientation, Monkey 1','Interpreter','latex')
c= colorbar;
title(c,'Average Firing Rate','Interpreter','latex');

hold on
line(xlim, [165,165], 'LineWidth', 2, 'Color', 'white','LineStyle','--');
%%
% Find Neuron Number for Each Array Element
Raw_data= load('data_monkey1_gratings');
SNR_threshold = 1.5; firing_rate_threshold=1.0;
[keepNeurons1, keepNeurons2]= find_good_neuron(Raw_data.data,data.S,SNR_threshold,firing_rate_threshold);

good_neuron= zeros(size(keepNeurons1));
[M, ~]= find(keepNeurons1==0);
indx= 1;

for i= 1:length(keepNeurons1)
    if isempty(find(M==i))
        good_neuron(i)=  keepNeurons2(indx);
        indx= indx+1;
    end
end

MAP_neurons= zeros(size(Raw_data.data.MAP));

for i=1:size(Raw_data.data.MAP,1)
    for j=1:size(Raw_data.data.MAP,2)
        index= find(Raw_data.data.CHANNELS(:,1) ==Raw_data.data.MAP(i,j));
        index(good_neuron(index)==0)=[]; %remove it if it is not good neuron
        [M, I]= max(Raw_data.data.SNR(index)); % consider the neuron with more SNR
        if ~isempty(I)
            MAP_neurons(i,j)= nnz(good_neuron(1:index(I)));
        else
            MAP_neurons(i,j)= NaN;
        end
    end
end
%% Pinwheel
clc

prefered_orinentation= zeros(1,num_neurons);
count= 1;
[M, I]= max(mean_fire_trial,[],2);

for i= 1:num_neurons
    prefered_orinentation(i)= orientation(I(i));
    if prefered_orinentation(i)>=180, prefered_orinentation(i)= (prefered_orinentation(i)-180); end
end

color= zeros(size(MAP_neurons));

for i=1:size(color,1)
    for j= 1:size(color,2)
        if ~isnan(MAP_neurons(i,j))
        color(i,j)= prefered_orinentation(MAP_neurons(i,j));
        else
           color(i,j)= nan; 
        end
    end
end

% show NaN array white.
figure
h=imagesc(color);
set(h, 'AlphaData', ~isnan(color))
title('Neurons Preferred Orientation, Monkey 1','Interpreter','latex');
colormap jet
colorbar
% ----------------------------------------------------------------
color(isnan(color))=0;
[X,Y] = meshgrid(1:10, 1:10);
[X2,Y2] = meshgrid(1:0.01:10, 1:0.01:10);

outData = interp2(X, Y, color, X2, Y2, 'linear');
figure
h=imagesc(outData);

set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', 1:size(X,2));
set(gca, 'YTickLabel', 1:size(X,1));
colormap jet
colorbar
Alpha= logical(ones(size(outData)));
for i=1:size(outData,1)*size(outData,2)
   if outData(i)== 0
       Alpha(i)= 0;
   end
end
set(h, 'AlphaData', logical(Alpha))
title('Neurons Preferred Orientation, Monkey 1','Interpreter','latex');
%% Noise Correlation
clc
R_sig= corrcoef(mean_fire_trial');

% Noise Correlation
Spike_count= zeros(num_neurons,num_stimulation*num_trials);
count= 1;

for i= 1:num_stimulation
    for j= 1:num_trials
        Spike_count(:,count)= sum(S(i).trial(j).spikes,2);
        count= count+1;
    end
    Spike_count(:,(i-1)*num_trials+1:i*num_trials)= zscore(Spike_count(:,(i-1)*num_trials+1:i*num_trials)')';
end

r_sc= corrcoef(Spike_count');


% Distance between electrodes
distance= zeros(num_neurons,num_neurons);
CH= Raw_data.data.CHANNELS(good_neuron==1,:);

for i= 1:num_neurons
    for j=1:num_neurons
        [x1, y1]= find(Raw_data.data.MAP== CH(i,1));
        [x2, y2]= find(Raw_data.data.MAP== CH(j,1));
        distance(i,j)= 0.4*sqrt((x1-x2)^2+(y1-y2)^2);
    end
end

%%
clc
% Noise Corr and Distance
R_sig2= tril(R_sig,-1);
bin= 0.25:0.5:4.25;
r_sc_select1= [];

for i= 1:length(bin)
        indx= find(R_sig2>0.5 & distance>= bin(i)-0.25 & distance< bin(i)+0.25);
        r_sc_select1(i)= mean(r_sc(indx));
        %error(i)= var(r_sc(indx));
end
error= (nanstd(r_sc_select1)./sqrt(length(r_sc_select1))).*ones(1,length(r_sc_select1));
figure
plot(bin,r_sc_select1,'LineWidth',8,'Color',[183 183 183]./255);

hold on
r_sc_select2= [];

for i= 1:length(bin)
        indx= find(R_sig2>0 & R_sig2<= 0.5 & distance>= bin(i)-0.25 & distance< bin(i)+0.25);
        r_sc_select2(i)= mean(r_sc(indx));
        %error(i)= var(r_sc(indx));
end
error= (nanstd(r_sc_select2)./sqrt(length(r_sc_select2))).*ones(1,length(r_sc_select2));
plot(bin,r_sc_select2,'LineWidth',6,'Color',[121 121 121]./255);

hold on
r_sc_select3= [];

for i= 1:length(bin)
        indx= find(R_sig2>=-0.5 & R_sig2< 0 & distance>= bin(i)-0.25 & distance< bin(i)+0.25);
        r_sc_select3(i)= mean(r_sc(indx));
        %error(i)= var(r_sc(indx));
end
error= (nanstd(r_sc_select3)./sqrt(length(r_sc_select3))).*ones(1,length(r_sc_select3));
plot(bin,r_sc_select3,'LineWidth',4,'Color',[94 94 94]./255);

hold on
r_sc_select= [];

for i= 1:length(bin)
        indx= find(R_sig2<-0.5 & distance>= bin(i)-0.25 & distance< bin(i)+0.25);
        r_sc_select(i)= mean(r_sc(indx));
        %error(i)= var(r_sc(indx));
end
error= (nanstd(r_sc_select)./sqrt(length(r_sc_select))).*ones(1,length(r_sc_select));
plot(bin,r_sc_select,'LineWidth',2,'Color','k'); hold on;


xlabel('Distance between electrodes (mm)','Interpreter','latex')
ylabel('Spike count correlation $(r_{sc})$','Interpreter','latex')
title('Monkey 1','Interpreter','latex')
xlim([0 5])
grid minor

hold all

errorbar(bin,r_sc_select1,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');
errorbar(bin,r_sc_select2,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');
errorbar(bin,r_sc_select3,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');
errorbar(bin,r_sc_select,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');

L= legend('$>0.5$','$0\quad to \quad 0.5$','$-0.5\quad to \quad 0$','$<-0.5$','Interpreter','latex');
title(L,'$R_{sig}$','Interpreter','latex')
%%
% Noise Corr and Signal Corr
distance2= tril(distance,-1);
bin= -0.75:0.25:0.75;
r_sc_select1= [];

for i= 1:length(bin)
        indx= find(distance2>=0 & distance2<1 & R_sig>= bin(i)-0.125 & R_sig< bin(i)+0.125);
        r_sc_select1(i)= mean(r_sc(indx));
end
error= (nanstd(r_sc_select1)./sqrt(length(r_sc_select1))).*ones(1,length(r_sc_select1));
figure
plot(bin,r_sc_select1,'LineWidth',8,'Color',[183 183 183]./255);

hold on
r_sc_select2= [];

for i= 1:length(bin)
        indx= find(distance2>=1 & distance2<2 & R_sig>= bin(i)-0.125 & R_sig< bin(i)+0.125);
        r_sc_select2(i)= mean(r_sc(indx));
end
error= (nanstd(r_sc_select2)./sqrt(length(r_sc_select2))).*ones(1,length(r_sc_select2));
plot(bin,r_sc_select2,'LineWidth',6,'Color',[121 121 121]./255);

hold on
r_sc_select3= [];

for i= 1:length(bin)
        indx= find(distance2>=2 & distance2<3 & R_sig>= bin(i)-0.125 & R_sig< bin(i)+0.125);
        r_sc_select3(i)= mean(r_sc(indx));
end
error= (nanstd(r_sc_select3)./sqrt(length(r_sc_select3))).*ones(1,length(r_sc_select3));
plot(bin,r_sc_select3,'LineWidth',4,'Color',[94 94 94]./255);

hold on
r_sc_select= [];

for i= 1:length(bin)
        indx= find(distance2>=3 & distance2<10 & R_sig>= bin(i)-0.125 & R_sig< bin(i)+0.125);
        r_sc_select(i)= mean(r_sc(indx));
end
error= (nanstd(r_sc_select)./sqrt(length(r_sc_select))).*ones(1,length(r_sc_select));
plot(bin,r_sc_select,'LineWidth',2,'Color','k');


xlabel('Orientation tuning similarity $(R_{signal})$','Interpreter','latex')
ylabel('Spike count correlation $(r_{sc})$','Interpreter','latex')
title('Monkey 1','Interpreter','latex')
xlim([-1 1])
grid minor

hold all

errorbar(bin,r_sc_select1,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');
errorbar(bin,r_sc_select2,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');
errorbar(bin,r_sc_select3,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');
errorbar(bin,r_sc_select,error,'CapSize',0.5,'LineWidth',2,'Color','k','LineStyle','none');

[L, hobj, ~, ~] = legend('$0-1 \quad mm$','$1-2 \quad mm$','$2-3 \quad mm$','$>3 \quad mm$','Interpreter','latex');
title(L,'Distance','Interpreter','latex')

%%
% r_sc, R_sig, distance
%r_sc2= tril(r_sc,-1);
R_sig2= tril(R_sig,-1);
distance2= tril(distance,-1);

bin1= 0.25:0.5:4.25; bin2= -1:0.25:1;
r_sc_select= [];

for i= 1:length(bin1)
    for j= 1:length(bin2)
        indx= find(R_sig2>= bin2(j)-0.125 & R_sig2< bin2(j)+0.125 & distance2>= bin1(i)-0.25 & distance2< bin1(i)+0.25);
        r_sc_select(i,j)= mean(r_sc(indx));
        %if ~isempty(indx), r_sc_select(i,j)= mean(r_sc2(indx)); end
    end
end

r_sc_select1= imgaussfilt(r_sc_select);
figure
h=imagesc('XData',bin1,'YData',bin2,'CData',r_sc_select1');
set(h, 'AlphaData', (~isnan(r_sc_select1))')
title('Monkey 1','Interpreter','latex')
ylabel('Orientation tuning similarity $(R_{signal})$','Interpreter','latex')
xlabel('Distance between electrodes (mm)','Interpreter','latex')
ylim([-1.1 1.1])
xlim([0 5])
colormap jet
c= colorbar;
h= ylabel(c,'Spike count correlation $(r_{sc})$','Rotation',270,'Interpreter','latex')
h.Position= [3  0.1350 0];

figure
h= pcolor(bin1,bin2,r_sc_select');
set(h, 'AlphaData', (~isnan(r_sc_select))')
h.FaceColor = 'interp'; axis square; set(h, 'EdgeColor', 'none');
title('Monkey 1','Interpreter','latex')
ylabel('Orientation tuning similarity $(R_{signal})$','Interpreter','latex')
xlabel('Distance between electrodes (mm)','Interpreter','latex')
ylim([-1.1 1.1])
xlim([0 5])
colormap jet
c= colorbar;
h= ylabel(c,'Spike count correlation $(r_{sc})$','Rotation',270,'Interpreter','latex')
h.Position= [3  0.1074 0];
%% Function

function [keepNeurons1, keepNeurons2]= find_good_neuron(data,S,SNR_threshold,firing_rate_threshold)
keepNeurons1 = data.SNR >= SNR_threshold;
% removed= [];
% removed= find(keepNeurons1==0);
%data.SNR < SNR_threshold

num_grats = length(S);
num_trials = length(S(1).trial);

for igrat = 1:num_grats
    for itrial = 1:num_trials
        S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons1,:);
    end
end

num_grats = length(S);
num_trials = length(S(1).trial);

mean_FRs = [];

for igrat = 1:num_grats
    for itrial = 1:num_trials
        mean_FRs = [mean_FRs sum(S(igrat).trial(itrial).spikes,2)/1.0];
    end
end

mean_FRs_gratings = mean(mean_FRs,2);
keepNeurons2 = mean_FRs_gratings >= firing_rate_threshold;
end


