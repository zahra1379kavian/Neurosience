function [] = plotRaster(spikeMat, tVec)
hold all;
for trialCount = 1:size(spikeMat,1)
    spikePos = tVec(spikeMat(trialCount, :));
    for spikeCount = 1:length(spikePos)
        if trialCount== 1
        plot([spikePos(spikeCount) spikePos(spikeCount)], ...
            [size(spikeMat,1)-trialCount-0.4 size(spikeMat,1)-trialCount+0.4], 'r','linewidth',0.8);
        elseif trialCount== 2
        plot([spikePos(spikeCount) spikePos(spikeCount)], ...
            [size(spikeMat,1)-trialCount-0.4 size(spikeMat,1)-trialCount+0.4], 'g','linewidth',0.8);
        elseif trialCount== 3
        plot([spikePos(spikeCount) spikePos(spikeCount)], ...
            [size(spikeMat,1)-trialCount-0.4 size(spikeMat,1)-trialCount+0.4], 'k','linewidth',0.8);
        end
    end
end
ylim([0 size(spikeMat, 1)+1]);
end