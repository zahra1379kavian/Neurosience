close all
clc
clear

% Homework 01
%% Integrate and Fire Neuron
% part a
fr = 100; tSim = 1; nTrials = 1000; dt = 1/1000;
[spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials, dt);
plotRaster(spikeMat, tVec);
xlabel('$time (s)$','Interpreter','latex')
title('Spike Train Using a Poisson Process')
%% Integrate and Fire Neuron
% part b
clc

spike_count = sum(spikeMat,2);

figure
histogram(spike_count,'Normalization','pdf');
title('spike count probability and spike count density');
ylabel('$probability$','Interpreter','latex')
xlabel('$spike number$','Interpreter','latex')

hold on

pd = fitdist(spike_count,'Poisson');
x = min(spike_count)-3:1:max(spike_count)+5;
y = poisspdf(x,pd.lambda);
plot(x,y,'LineWidth',2)

%% Integrate and Fire Neuron
% part c
clc

ISI = ISI_calculator(spikeMat, nTrials);

figure
histogram(ISI,'Normalization','pdf');
title('ISI histogram and ISI density');
ylabel('$probability$','Interpreter','latex')
xlabel('$Inter Spike Interval$','Interpreter','latex')

hold on

x = 0:0.001:max(ISI);
pd = fitdist(ISI','Exponential');
pd.mu = pd.mu*0.01;
f = 2.09.*pd.mu.*exp(-pd.mu.*x);
plot(x,f,'linewidth',2);
xlim([0 max(ISI)])

%% Renewal Method
% part a
fr = 100; tSim = 1; nTrials = 10000; dt = 1/1000; k = 5;
[spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials, dt);
spikeMat_renewal = renewal_spike_gen(spikeMat, nTrials, k);
spikeMat_renewal = logical(spikeMat_renewal);

plotRaster(spikeMat_renewal, tVec)
xlabel('$time (s)$','Interpreter','latex')
title('Spike Train Using a Renewal Process')
%% Renewal Method
% part b
clc

spike_count = sum(spikeMat_renewal,2);

figure
histogram(spike_count,'Normalization','pdf');
title('spike count probability and spike count density','Interpreter','latex');
ylabel('$probability$','Interpreter','latex')
xlabel('$spike number$','Interpreter','latex')

hold on

%pd = fitdist(spike_count,'Gamma');
x = min(spike_count)-3:1:max(spike_count)+10;
%y = gampdf(x,pd.a,pd.b);
lambda = 5;
y = poisspdf(x-15,lambda);
plot(x,y,'LineWidth',2)


%% Renewal Method
% part c
clc

ISI_renew = ISI_calculator(spikeMat_renewal, nTrials);

figure
histogram(ISI_renew,'Normalization','pdf');
title('ISI histogram and ISI density','Interpreter','latex');
ylabel('$probability$','Interpreter','latex')
xlabel('$Inter Spike Interval$','Interpreter','latex')

hold on

x = 0:0.001:max(ISI_renew);
pd = fitdist(ISI_renew','Gamma');
y = gampdf(x,pd.a,pd.b);
plot(x,y,'linewidth',2);

%%
% part d
cv = CV_func(ISI)
cv_ISI = CV_func(ISI_renew)
%%
%part f
fr = 100; tSim = 3; nTrials = 10000; dt = 1/1000; 

for k = 1:30
[cv(k) isi(k)]= renewal_ISI_Cv(fr, tSim, nTrials, dt, k);
end

figure
scatter(k,cv)
%% Refactory Period
% part g
clc

T = 0:0.0001:0.03; fr = 1./T; nTrials = 1000;
k = [1,4,51]; dt = 1/1000; tSim = 5; trefact = 1;

hold all

for j = 1:length(k)
    for i = 1:length(fr)
        [spikeMat, tVec] = poissonSpikeGen(fr(i), tSim, nTrials, dt);
        spikeMat_renewal = renewal_spike_gen(spikeMat, nTrials, k(j));
        ISI_renew = ISI_calculator(spikeMat_renewal, nTrials);
        ISI_renew(ISI_renew>trefact);
        cv(j,i) = CV_func(ISI_renew);
    end
    
    scatter(T,cv(j,:))
end

text(T(150),cv(1,150)-0.1,'N_{th}=1, t_{0}=1 msec')
text(T(150),cv(2,150)-0.1,'N_{th}=4, t_{0}=1 mesc')
text(T(150),cv(3,150)-0.1,'N_{th}=51, t_{0}=1 mesc')

cv_theory = 1./sqrt(k);
for j = 1:length(k)
    plot(T,ones(1,length(T)).*cv_theory(j),'Color','black');
end

text(T(150),cv_theory(1),'N_{th}=1, t_{0}=0')
text(T(150),cv_theory(2),'N_{th}=4, t_{0}=0')
text(T(150),cv_theory(3),'N_{th}=51, t_{0}=0')

xlabel('\Delta_{t} (msec)')
ylabel('C_{v}')

%%
function [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials, dt)
nBins = floor(tSim/dt);
spikeMat = rand(nTrials, nBins) < fr*dt;
tVec = 0:dt:tSim-dt;
end

function [] = plotRaster(spikeMat, tVec)
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

function cv = CV_func(x)
cv = std(x)/mean(x);
end

function ISI = ISI_calculator(spikeMat, nTrials)
s = 1;
ISI = [];
for i = 1:nTrials
    a = diff(find(spikeMat(i,:)));
    ISI = [ISI a];
end
end

function spikeMat_renewal = renewal_spike_gen(spikeMat, nTrials, k)
spikeMat_renewal = zeros(size(spikeMat));

for j = 1:nTrials
    x = find(spikeMat(j,:));
    for i = k:k:length(x)
        spikeMat_renewal(j,x(i)) = 1;
    end
end
end

function [cv,isi] = renewal_ISI_Cv(fr, tSim,nTrials, dt, k)
[spikeMat, ~] = poissonSpikeGen(fr, tSim, nTrials, dt);
spikeMat_renewal = renewal_spike_gen(spikeMat, nTrials, k);
spikeMat_renewal = logical(spikeMat_renewal);

ISI_renew = ISI_calculator(spikeMat_renewal, nTrials);
cv = CV_func(ISI_renew);
isi = mean(ISI_renew);
end