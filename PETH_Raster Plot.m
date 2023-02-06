close all
clc
clear

data = load('Q2_data');

dt = 1/2;
tSim = 200;
tVec = -50:dt:tSim-dt;

d = data.trials;

figure
Raster_Plot(d(1,:),tVec)
figure
Raster_Plot(d(1:20,:),tVec)
figure
Raster_Plot(d,tVec)
%%
clc

width = 10;
PETH(width,d);
width = 2.5;
PETH(width,d)
width = 25;
PETH(width,d)


function Raster_Plot(d,tVec)
spikeMat =  logical(d);

hold all;
for trialCount = 1:size(spikeMat,1)
    spikePos = tVec(spikeMat(trialCount, :));
    for spikeCount = 1:length(spikePos)
        plot([spikePos(spikeCount) spikePos(spikeCount)], ...
            [trialCount-0.4 trialCount+0.4], 'k');
    end
end
ylim([0 size(spikeMat, 1)+1]);
hold on

xline(0,'--b','LineWidth',2)
xlabel('Time (ms)');
ylabel('Trial Number');
end

function PETH(width,d)
p = zeros(100,250/width);
s = 1;
for k = 1:100
for i = 1:2*width:size(d,2)
    p(k,s) = p(k,s) + nnz(d(k,i:i+(2*width-1)))*100;
    s = s+1;
end
s = 1;
end
figure
firingRate = mean(p);
edges = [-50:width:200-width];
bar(edges,firingRate)
hold on
xline(0,'--b','LineWidth',2)
figure
plot(edges,firingRate)
title('firing rate')
end