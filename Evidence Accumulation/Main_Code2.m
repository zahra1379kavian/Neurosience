clc
close all
clear

% part 2
% Question 1

MT_p_values= [0.6 0.4];
LIP_weights= [0.5 -0.1]; % [0.3 -0.1]
LIP_threshold= 100;
Evidence_thr= 25;

[LIP_event_times,count,MT_spike,LIP_event,p_lip,dt]= LIP_activity(MT_p_values,LIP_weights,LIP_threshold,Evidence_thr);

clc
subplot(1,2,1)
tSim= (count-1)*dt;
tVec = 0:dt:tSim-dt;
spikeMat= [MT_spike';LIP_event];
plotRaster(spikeMat, tVec);
xline(LIP_event_times(1),'--b')
ylabel('Event Times','interpreter','latex')
xlabel('Time','interpreter','latex')

subplot(1,2,2)
stairs(tVec,p_lip,'linewidth',1.5)
yline(Evidence_thr,'--k')
hold on
scatter(LIP_event_times(1),p_lip(LIP_event_times(1)/dt),'fill')
ylabel('Dicision Value','interpreter','latex')
xlabel('Time','interpreter','latex')
sgtitle('MT-LIP Interaction','interpreter','latex')

%%
clc
clear
% Question 2

direction= linspace(-pi,pi,100);
mu= 0; sigma= 0.5;
tuning_cutve1 = normpdf(direction,mu,sigma); %MT1
mu= 0; sigma= 0.6;
tuning_cutve2 = normpdf(direction,mu,sigma); %MT2

figure
subplot(1,2,1)
plot(direction,tuning_cutve1,'linewidth',1.5)
title('Tuning Curve Neuron 1','interpreter','latex')
xlabel('firing rate','interpreter','latex')
grid minor
subplot(1,2,2)
plot(direction,tuning_cutve2,'linewidth',1.5)
title('Tuning Curve Neuron 2','interpreter','latex')
grid minor
xlabel('firing rate','interpreter','latex')
%%

LIP_weights1= [0.5 -0.4]; % [0.9 -0.1] %[0.5 -0.4]
LIP_weights2= [1 -0.01]; % [0.8 -0.05] %[1 -0.01]
LIP_threshold= 100; %100
Evidence_thr= 25; %25

dt= 0.01;
rate= 0;
N= [0 0]; % plus is first, minus is second
count1= 1; count2= 1;
LIP_event_times1= [];
LIP_event_times2= [];
MT_spike1= [];
MT_spike2= [];
LIP_event1= [];
LIP_event2= [];
p_lip1= [];
p_lip2= [];

for i=1:2
    N= [0 0]; t= 0; rate=0;
    while rate<LIP_threshold
       stimulus= randi(length(direction),1);
       firing_rate_MT= [tuning_cutve1(stimulus) tuning_cutve2(stimulus)];
       dN= rand(1,2)< firing_rate_MT;
       N= N+ dN;
        
        if i== 1
            MT_spike1(i,count1,:)= dN(1);
            MT_spike2(i,count1,:)= dN(2);
            
            p_lip1(count1)= sum(N.*LIP_weights1);
            
            LIP_event1(count1)= p_lip1(count1)> Evidence_thr;
            if LIP_event1(count1) == 1
                LIP_event_times1 = [LIP_event_times1 t];
            end
            
            % check LIP mean rate for last M spikes
            M = 100;
            if length(LIP_event_times1)>=M
                rate = M/(t-LIP_event_times1(length(LIP_event_times1)-M+1));
            end
            
            count1= count1+1;
            t= (count1-1)*dt;
            
        elseif i==2
            MT_spike1(i,count2,:)= dN(1);
            MT_spike2(i,count2,:)= dN(2);
            
            p_lip2(count2)= sum(N.*LIP_weights2);
            LIP_event2(count2)= p_lip2(count2)> Evidence_thr;
            if LIP_event2(count2) == 1
                LIP_event_times2 = [LIP_event_times2 t];
            end
            
            % check LIP mean rate for last M spikes
            M = 100;
            if length(LIP_event_times2)>=M
                rate = M/(t-LIP_event_times2(length(LIP_event_times2)-M+1));
            end
            count2= count2+1;
            t= (count2-1)*dt;
        end
        
    end
end


figure
subplot(1,2,1)
tSim= (count1-1)*dt;
tVec = 0:dt:tSim-dt;
spikeMat= [squeeze(MT_spike1(1,1:count1-1,:));squeeze(MT_spike2(1,1:count1-1,:));LIP_event1];
plotRaster(logical(spikeMat), tVec);
xline(LIP_event_times1(1),'--b')
ylabel('Event Times','interpreter','latex')
xlabel('Time','interpreter','latex')

subplot(1,2,2)
stairs(tVec,p_lip1,'linewidth',1.5)
yline(Evidence_thr,'--k')
hold on
scatter(LIP_event_times1(1),p_lip1(round(LIP_event_times1(1)/dt)),'fill')
ylabel('Dicision Value','interpreter','latex')
xlabel('Time','interpreter','latex')
sgtitle('First LIP Neuron','interpreter','latex')

figure
subplot(1,2,1)
tSim= (count2-1)*dt;
tVec = 0:dt:tSim-dt;
spikeMat= [squeeze(MT_spike1(2,1:count2-1,:));squeeze(MT_spike2(1,1:count2-1,:));LIP_event2];
plotRaster(logical(spikeMat), tVec);
xline(LIP_event_times2(1),'--b')
ylabel('Event Times','interpreter','latex')
xlabel('Time','interpreter','latex')

subplot(1,2,2)
stairs(tVec,p_lip2,'linewidth',1.5)
yline(Evidence_thr,'--k')
hold on
scatter(LIP_event_times2(1),p_lip2(round(LIP_event_times2(1)/dt)),'fill')
ylabel('Dicision Value','interpreter','latex')
xlabel('Time','interpreter','latex')
sgtitle('Second LIP Neuron','interpreter','latex')


function [] = plotRaster_2(spikeMat, tVec)
hold all;
for trialCount = 1:size(spikeMat,1)
    spikePos = tVec(spikeMat(trialCount, :));
    for spikeCount = 1:length(spikePos)
        plot([spikePos(spikeCount) spikePos(spikeCount)], ...
            [trialCount-0.4 trialCount+0.4], 'k');
    end
end
ylim([0 size(spikeMat, 1)+1]);
end
