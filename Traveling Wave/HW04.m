clc
close all
clear
% Homework 04
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data
load('ArrayData');
load('CleanTrials');
fs= round(1/(Time(2)-Time(1)));

bad_trial= ones(1,size(chan(1).lfp,2));
bad_trial(Intersect_Clean_Trials)= 0;
bad_trial= find(bad_trial==1);

% pre-process
for i= 1:length(chan)
    chan(i).lfp(:,bad_trial)= [];
end
%% Data Observation
% plot pwelch each trial and their average
electrode_num= 1;

data= chan(1).lfp;
[pxx,f]= pwelch(data,fs);

figure
subplot(1,2,1)
for i= 1:size(pxx,2)
    plot(log(f),log(pxx(:,i)),'color',[200 191 231]./255)
    hold on
end
hold on
plot(log(f),log(mean(pxx,2)),'linewidth',2,'color','red')
xlabel('log(frequency)','Interpreter','latex')
ylabel('log(power)','Interpreter','latex')
title('power spectrum each trials and their mean, electrode 1','Interpreter','latex')

data= chan(40).lfp;
[pxx,f]= pwelch(data,fs);


subplot(1,2,2)
for i= 1:size(pxx,2)
    plot(log(f),log(pxx(:,i)),'color',[200 191 231]./255)
    hold on
end
hold on
plot(log(f),log(mean(pxx,2)),'linewidth',2,'color','red')
xlabel('log(frequency)','Interpreter','latex')
ylabel('log(power)','Interpreter','latex')
title('power spectrum each trials and their mean, electrode 40','Interpreter','latex')
%% Data Observation
clc
select_trial= 100;

figure

for i= 1:48
    plot(Time,squeeze(chan(i).lfp(:,select_trial))+((48-i)+4*i)*200,'linewidth',2)
    hold on
end
xlim([-1.2 2])
title('LFP Signal, All Electrode, Trial Number= 100','Interpreter','latex')
xlabel('Time (ms)')

%% Dominant Frequency
clc
% Concat all trial electrode - pspectrum

data= chan(electrode_num).lfp; LFP_sig= [];
for i= 1:size(data,2)
    LFP_sig= cat(2,LFP_sig,data(:,i)');
end

[p,f,y_noise,power_normalize]= dePinkNoise_func(LFP_sig,fs);

figure
plot(f,smooth(p),'color',[0 162 232]./255,'LineWidth',1.5);
hold on
plot(f,y_noise,'black','LineWidth',2)
hold on
plot(f,smooth(power_normalize),'color',[71,220,116]./255,'LineWidth',1.5)
hold on
title('Identify Dominant Frequency','Interpreter','latex')

[dominant_freq,I,f0]= Domain_freq_func(f,power_normalize);

tmp= smooth(power_normalize);
plot(dominant_freq,tmp(I+length(f0)),'r*')
legend('PowerSpectrum','pink noise','normalized power spectrum','Interpreter','latex');
xlabel('Frequency (Hz)','Interpreter','latex')
ylabel('log (power)','Interpreter','latex')
xlim([0 100])
%% Dominant Frequency
% Average over trials_spectrum - Morlet wavelets
clc
electrode_num= 1; wt=[];
data= chan(electrode_num).lfp;

[wt_mean,f,y_noise,power_normalize]= dePinkNoise_func_wavelete(data,fs);
% -----------------------------------------------------------------------
figure
plot(f,wt_mean,'color',[0 162 232]./255,'LineWidth',1.5)

hold on
plot(f,y_noise,'black','LineWidth',2)

hold on
plot(f,power_normalize,'color',[71,220,116]./255,'LineWidth',1.5)

hold on
plot(f,ones(size(f)),'k--')

hold on
[M, I]= max(power_normalize);
dominant_freq= f(I);

plot(dominant_freq,M,'r*')

legend('PowerSpectrum','pink noise','normalized power spectrum','Interpreter','latex','Location','southwest');
xlabel('Frequency (Hz)','Interpreter','latex')
ylabel('log (power)','Interpreter','latex')
title('Identify Dominant Frequency','Interpreter','latex')
%% Dominant Frequency
% Average over trials_spectrum - pspectrum
clc;

window= 100; noverlap= 50; fft= 200; electrode_num= 1;
data= chan(electrode_num).lfp; p=[]; y_noise=[]; power_normalize=[];

for i= 1:size(data,2)
    LFP_sig= data(:,i);
    [p(:,i),f,y_noise(:,i),power_normalize(:,i)]= dePinkNoise_func(LFP_sig,fs);
end

p= mean(p,2); y_noise= mean(y_noise,2); power_normalize= mean(power_normalize,2);

figure
plot(f,p,'color',[0 162 232]./255,'LineWidth',1.5);
hold on
plot(f,y_noise,'black','LineWidth',2)
hold on
plot(f,power_normalize,'color',[71,220,116]./255,'LineWidth',1.5)
hold on
title('Identify Dominant Frequency','Interpreter','latex')

[dominant_freq,I,f0]= Domain_freq_func(f,power_normalize);
tmp= smooth(power_normalize);
plot(dominant_freq,tmp(I+length(f0)),'r*')
legend('PowerSpectrum','pink noise','normalized power spectrum','Interpreter','latex');
xlabel('Frequency (Hz)','Interpreter','latex')
ylabel('log (power)','Interpreter','latex')
xlim([0 100])

%% Cluster Electrodes
% for all electrod - pspectrum
clc
for j= 1:length(chan)
    data= chan(j).lfp;
    for i= 1:size(data,2)
        [p(:,i),f]= pspectrum(data(:,i),fs);
    end
    
    p(:,j)= mean(p,2);
    mdlr = fitlm(f,p(:,j));
    y_noise= mdlr.Coefficients.Estimate(2).*f+mdlr.Coefficients.Estimate(1);
    power_normalize= p(:,j)-y_noise;
    
    dominant_freq(j)= Domain_freq_func(f,power_normalize);
end


figure
h= histogram(dominant_freq)
h.FaceColor= 'red';
h.EdgeColor= 'red';
title('Dominant Frequency','Interpreter','latex')
xlabel('Frequency (Hz)','Interpreter','latex')
ylabel('Counts of electrodes','Interpreter','latex')

color_data= NaN*ones(size(ChannelPosition));
for i= 1:4
    group= find(dominant_freq>=h.BinEdges(i) & dominant_freq<h.BinEdges(i+1));
    for j= 1:length(group)
        [r, c]= find(ChannelPosition==group(j));
        color_data(r,c)= (h.BinEdges(i)+h.BinEdges(i+1))/2;
    end
end

figure
heatmap(color_data);
colormap jet
title('Group Data Base on Dominant Frequency')
%% Cluster Electrode
% pspectrum
clc
xvalues= 1:size(ChannelPosition,2); yvalues= 1:size(ChannelPosition,1); cdata= zeros(size(ChannelPosition));
for i= 1:size(ChannelPosition,1)
    for j= 1:size(ChannelPosition,2)
        if isnan(ChannelPosition(i,j))
            cdata(i,j)= NaN;
        else
            cdata(i,j)=  dominant_freq(ChannelPosition(i,j));
        end
    end
end
figure
heatmap(xvalues,yvalues,cdata);
title('Electrodes Dominant Frequency')
colormap summer

%% Cluster Electrode
% for all electrod - wavelete
clc
wt_mean=[]; y_noise=[]; power_normalize=[];

for j= 1:length(chan)
    data= chan(j).lfp;
    [wt_mean(j,:),f,y_noise(:,j),power_normalize(:,j)]= dePinkNoise_func_wavelete(data,fs);
    
    [M, I]= max(power_normalize(:,j));
    dominant_freq(j)= f(I);
end

figure
h= histogram(dominant_freq);
h.FaceColor= 'red';
h.EdgeColor= 'red';
xlabel('Dominant Frequency','Interpreter','latex')

%% Cluster Electrode
% wavelete
clc
xvalues= 1:size(ChannelPosition,2); yvalues= 1:size(ChannelPosition,1); cdata= zeros(size(ChannelPosition));
for i= 1:size(ChannelPosition,1)
    for j= 1:size(ChannelPosition,2)
        if isnan(ChannelPosition(i,j))
            cdata(i,j)= NaN;
        else
            cdata(i,j)=  dominant_freq(ChannelPosition(i,j));
        end
    end
end
figure
heatmap(xvalues,yvalues,cdata);
title('Electrodes Dominant Frequency')
colormap summer
%% STFT
clc
p_raw=[]; p_denoise= []; 

for electrode_num= 1:length(chan)
    data= chan(electrode_num).lfp;
    for i= 1:size(data,2)
        [s,f,t]= stft(data(:,i),fs,'Window',kaiser(50,5),'OverlapLength',40,'FFTLength',200);
        s= s(end/2+1:end,:);
        f= f(end/2+1:end);
        signal_power= log10(abs(s).^2);
        
        p_raw(electrode_num,i,:,:)= signal_power;
    end
end


p1= squeeze(mean(squeeze(mean(p_raw,2)),1));

subplot(1,2,1)
pcolor(t-1,f,abs(p1).^2)
colormap jet
shading interp
xline(0,'--','linewidth',2)
ylim([1 70])
xlabel('Time (ms)','Interpreter','latex')
ylabel('Frequency (Hz)','Interpreter','latex')
colorbar


s_normalize= [];
for j= 1:size(p1,2)
    mdlr = fitlm(f,p1(:,j));
    y_noise= mdlr.Coefficients.Estimate(2).*f+mdlr.Coefficients.Estimate(1);
    s_normalize(:,j)= p1(:,j)-y_noise;
end

subplot(1,2,2)
pcolor(t-1,f,s_normalize)
shading interp
colormap jet
xline(0,'--','linewidth',2)
ylim([1 70])
xlabel('Time (ms)','Interpreter','latex')
ylabel('Frequency (Hz)','Interpreter','latex')
colorbar
%% Wavelet

clc
p_raw=[]; p_denoise= []; %electrode_num=1; s_normalize=[];

for electrode_num= 1:length(chan)
    data= chan(electrode_num).lfp;
    for i= 1:size(data,2)
        [wt,f] = cwt(data(:,i),fs);
        signal_power= log10(abs(wt).^2);
        
        p_raw(electrode_num,i,:,:)= signal_power;
    end
end

p1= squeeze(mean(squeeze(mean(p_raw,2)),1));

subplot(1,2,1)
pcolor(Time,f,p1)
colormap jet
shading interp
xline(0,'--','linewidth',2)
ylim([1 70])
xlabel('Time (ms)','Interpreter','latex')
ylabel('Frequency (Hz)')
colorbar



s_normalize= [];
for j= 1:size(p1,2)
    mdlr = fitlm(f,p1(:,j));
    y_noise= mdlr.Coefficients.Estimate(2).*f+mdlr.Coefficients.Estimate(1);
    s_normalize(:,j)= p1(:,j)-y_noise;
end

subplot(1,2,2)
pcolor(Time,f,s_normalize)
shading interp
colormap jet
xline(0,'--','linewidth',2)
ylim([1 45])
xlabel('Time (ms)','Interpreter','latex')
ylabel('Frequency (Hz)')
colorbar

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Butterworth Filter
clc
order= 2; low_f= 11; high_f= 13;
[b,a] = butter(order,[low_f high_f]./fs*2,'bandpass');
figure
freqz(b,a,1000,fs)

%% Filter Signal
% pwelch
clc
bandwidth= 1.5; order= 2;
LFP_signal= zeros([length(chan),size(chan(1).lfp)]); LFP_signal_filtered= zeros([length(chan),size(chan(1).lfp)]);

for i= 1:length(chan)
    LFP_signal(i,:,:)=  chan(i).lfp;
end

for i= 1:size(ChannelPosition,1)
    for j= 1:size(ChannelPosition,2)
        if ~isnan(ChannelPosition(i,j))
            fc= color_data(i,j);
            [b,a] = butter(order,[fc-bandwidth fc+bandwidth]./fs*2,'bandpass');
            LFP_signal_filtered(ChannelPosition(i,j),:,:) = filtfilt(b,a,squeeze(LFP_signal(ChannelPosition(i,j),:,:)));
        end
    end
end


%% compare signal before and after filtering
% pwelch

p= zeros(length(chan),101); window= 100; noverlap= 50; fft= 200;

for i= 1:size(LFP_signal,1)
    for j= 1:size(LFP_signal,3)
        [pxx,f] = pwelch(LFP_signal(i,:,j),window,noverlap,fft,fs);
        p(i,:)= p(i,:)+10*log(pxx');
    end
end

p_filt= zeros(length(chan),101);
for i= 1:size(LFP_signal_filtered,1)
    for j= 1:size(LFP_signal_filtered,3)
        [pxx,f] = pwelch(LFP_signal_filtered(i,:,j),window,noverlap,fft,fs);
        p_filt(i,:)= p_filt(i,:)+10*log(pxx');
    end
end


subplot(1,2,1)
ch= 1:48;
pcolor(ch,f,p'./size(chan(1).lfp,2));
shading interp
colormap jet
xlabel('channel','Interpreter','latex')
ylabel('frequency (Hz)','Interpreter','latex')
title('Raw Signal','Interpreter','latex')
colorbar


subplot(1,2,2)
pcolor(ch,f,p_filt'./size(chan(1).lfp,2));
shading interp
colormap jet
xlabel('channel','Interpreter','latex')
ylabel('frequency (Hz)','Interpreter','latex')
title('Filtered Signal','Interpreter','latex')
c= colorbar;
h= ylabel(c,'Average Power Spectrum','Rotation',270,'Interpreter','latex')
h.Position= [2.7578 6.8270 0];
h.FontSize= 12;

%%  Instantaneous Phase
clc

sig_angle= zeros(size(LFP_signal_filtered));

for i= 1:size(LFP_signal_filtered,1)
    sig_angle(i,:,:)= angle(hilbert(squeeze(LFP_signal_filtered(i,:,:))));
end

electrode_phase= zeros([size(ChannelPosition), size(sig_angle,[2,3])]);

for i= 1:size(ChannelPosition,1)
    for j= 1:size(ChannelPosition,2)
        if ~isnan(ChannelPosition(i,j))
            electrode_phase(i,j,:,:)= squeeze(sig_angle(ChannelPosition(i,j),:,:));
        else
            electrode_phase(i,j,:,:)= NaN;
        end
    end
end

%% Cos (phase) During Different Time Points
clc
select_trial= 100; % 259,50

figure;
colormap hot
for i= 1:size(electrode_phase,3)
    cla reset;
    imagesc(1:5,1:10,cos(squeeze(electrode_phase(:,:,i,select_trial))));
    %pcolor(cos(squeeze(electrode_phase(:,:,i,select_trial))))
    %shading interp
    title("Time: "+Time(i)+" s",'Interpreter','latex')
    drawnow
    thisFrame = getframe(gcf);
    myMovie(i) = thisFrame;
end

% %Write the videos
% %writerObj = VideoWriter("plane wave_shading");
% writerObj = VideoWriter("plane wave");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);
%% LFP Signal with local maximum
clc
select_trial= 400; %300

figure
for i= 1:48
    plot(Time,squeeze(LFP_signal_filtered(i,:,select_trial))+((48-i)+2*i)*200)
    hold on
    [pks,locs] = findpeaks(squeeze(LFP_signal_filtered(i,241:350,select_trial)));
    plot((locs(1))/fs,squeeze(LFP_signal_filtered(i,locs(1),select_trial))+((48-i)+2*i)*200,'*r')
end
xlim([-0.5 0.5])
xline(0,'--k','linewidth',4)
xline(0.03,'r','linewidth',1)
xline(0.125,'r','linewidth',1)
xlabel('Time (s)','Interpreter','latex')
ylabel('Voltage','Interpreter','latex')
title("Trial Number: "+int2str(select_trial),'Interpreter','latex')

%% Some frame of the wave
clc
t= round(0.03*fs:0.125*fs);
figure
for i= 1:length(t)
    subplot(4,5,i)
    imagesc(myMovie(241+t(i)).cdata);
    if i==1
        title("Time: "+Time(241+t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241+t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end


figure
for i= 1:length(t)
    subplot(4,5,i)
    imagesc(myMovie(241-t(i)).cdata);
    if i==1
        title("Time: "+Time(241-t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241-t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end
%% Phase Contourf
clc

figure
for i= 1:length(t)
    subplot(4,5,i)
    phi= squeeze(electrode_phase(:,:,241-t(i),select_trial));
    contourf(phi)
    shading interp
    colorbar
    colormap gray
    if i==1
        title("Time: "+Time(241-t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241-t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end

figure
for i= 1:length(t)
    subplot(4,5,i)
    phi= squeeze(electrode_phase(:,:,241+t(i),select_trial));
    contourf(phi)
    shading interp
    colorbar
    colormap gray
    if i==1
        title("Time: "+Time(241+t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241+t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end
%% LFP Signal with local maximum  - Average all trials
clc

figure
for i= 1:48
    data_prim= squeeze(mean(LFP_signal_filtered(i,:,:),3));
    plot(Time,data_prim+((48-i)+2*i)*30)
    hold on
end
xlim([-0.5 0.5])
xline(0,'--k','linewidth',4)
xline(0.03,'r','linewidth',1)
xline(0.125,'r','linewidth',1)
xlabel('Time (s)','Interpreter','latex')
ylabel('Voltage','Interpreter','latex')
title("Avergae Over All Trials",'Interpreter','latex')

%% Average all trials
figure
colormap hot
for i= 1:size(electrode_phase,3)
    cla reset;
    data_prim= squeeze(mean(squeeze(electrode_phase(:,:,i,:)),3));
    pcolor(cos(data_prim))
    shading interp
    title("Time: "+Time(i)+" s",'interpreter','latex')
    drawnow
    thisFrame = getframe(gcf);
    myMovie(i) = thisFrame;
end

% %Write the videos
% writerObj = VideoWriter("plane wave_shading_average_all_trials");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);

%% Some frame of the wave - Average all trials
clc
t= round(0.03*fs:0.125*fs);
figure
for i= 1:length(t)
    subplot(4,5,i)
    imagesc(myMovie(241+t(i)).cdata);
    if i==1
        title("Time: "+Time(241+t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241+t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end


figure
for i= 1:length(t)
    subplot(4,5,i)
    imagesc(myMovie(241-t(i)).cdata);
    if i==1
        title("Time: "+Time(241-t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241-t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end

%% Phase Gradient Directionality (PGD), Velocity Direction, Speed
clc
px=[]; py=[];

for j= 1:size(electrode_phase,4) % for each trial
    for i= 1:size(electrode_phase,3) % for each time
        instantaneous_phase= squeeze(electrode_phase(:,:,i,j));
        [px(i,j,:,:),py(i,j,:,:)]= gradient(instantaneous_phase);
        PGD(j,i)= norm(nanmean(squeeze(px(i,j,:,:)),'all')+1i.*nanmean(squeeze(py(i,j,:,:)),'all'))./...
            nanmean(sqrt(squeeze(px(i,j,:,:)).^2+squeeze(py(i,j,:,:)).^2),'all');
        velocity_direction(j,i)= nanmean(squeeze(px(i,j,:,:)+1i*py(i,j,:,:)),'all');
    end
    temp= zeros(1,size(electrode_phase,3));
    temp(1,2:end)= abs(squeeze(nanmean(nanmean(squeeze(diff(electrode_phase(:,:,:,j),1,3).*200),1),2)));
    speed(j,:)= temp'./...
        nanmean(nanmean(sqrt(squeeze(px(:,j,:,:)./0.04).^2+squeeze(py(:,j,:,:)./0.04).^2),2),3);
end


%% PGD Results
clc
figure
% Mean PGD over time
plot(Time,mean(PGD,1),'linewidth',2)
title('Average PGD Across Trials','interpreter','latex')
xlabel('Time (s)','interpreter','latex')
ylabel('Mean PGD','interpreter','latex')
xlim([-1 2])
xline(0,'--k','linewidth',1)

figure;
% Mean PGD over trials
subplot(1,2,1)
polarhistogram(mean(PGD>=0.5,2),'EdgeColor','k','FaceColor','k')
subplot(1,2,2)
histogram(rad2deg(mean(PGD>=0.5,2)),'EdgeColor','k','FaceColor','k')
sgtitle("Average PGD Over Time, $PGD \geq 0.5$",'interpreter','latex')
xlabel('Average PGD (deg)','interpreter','latex')
ylabel('PGD Count','interpreter','latex')
%%
figure
% a trial PGD over time
selected_trial= 100;

for i= 1:size(PGD,2)
    if i<241 & PGD(selected_trial,i)>=0.5
        bar(PGD(selected_trial,i),'EdgeColor','k','FaceColor','k')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','k')
    elseif i<241 & PGD(selected_trial,i)<0.5
        bar(PGD(selected_trial,i),'EdgeColor','red','FaceColor','red')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','k')
    elseif i>=241 & PGD(selected_trial,i)>=0.5
        bar(PGD(selected_trial,i),'EdgeColor','b','FaceColor','b')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','b')
    elseif i>=241 & PGD(selected_trial,i)<0.5
        bar(PGD(selected_trial,i),'EdgeColor','red','FaceColor','red')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','b')
    end
    ylim([min(PGD(selected_trial,:)) max(PGD(selected_trial,:))])
    pause(0.1)
    
    thisFrame = getframe(gcf);
    myMovie(i) = thisFrame;
end

% %Write the videos
% writerObj = VideoWriter("PGD a trial over time");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);

%% Velocity Direction
clc
selected_trial= 100; velocity_direction=[];

figure
for i= 1:size(electrode_phase,3)
    instantaneous_phase= squeeze(electrode_phase(:,:,i,selected_trial));
    [px1,py1]= gradient(instantaneous_phase);
    contour(1:10,1:5,instantaneous_phase,'LineWidth',2)
    hold on
    h1= quiver(1:10,1:5,px1,py1);
    set(h1,'LineWidth',3)
    if i<241
        set(h1,'Color','k')
    else
        set(h1,'Color','red')
    end
    hold off
    xlim([1 10]); ylim([1 5]);
    title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','k')
    drawnow();
    pause(0.1)
    thisFrame = getframe(gcf);
    myMovie(i) = thisFrame;
end

% %Write the videos
% writerObj = VideoWriter("Velocity Direction Over Time");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);

%% some frame of velocity direction video
t= round(0.03*fs:0.125*fs);
figure
for i= 1:length(t)
    subplot(4,5,i)
    imagesc(myMovie(241+t(i)).cdata);
    if i==1
        title("Time: "+Time(241+t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241+t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end

figure
for i= 1:length(t)
    subplot(4,5,i)
    imagesc(myMovie(241-t(i)).cdata);
    if i==1
        title("Time: "+Time(241-t(i))+" s",'interpreter','latex')
        ylabel('Array dimension','interpreter','latex')
    else
        title(Time(241-t(i)),'interpreter','latex')
    end
    set(gca,'XTick',[],'YTick',[])
end
%% speed result
clc

figure
histogram(speed(PGD>0.5),'EdgeColor','k','FaceColor','k')
xlim([0 200])
xlabel('speed (cm/s)','interpreter','latex')
ylabel('Speed Count','interpreter','latex')
title("Speed acrosee trials and times for $PGD\geq0.5$",'interpreter','latex')

figure
selected_trial= 100;
histogram(speed(selected_trial,PGD(selected_trial,:)>0.5),'EdgeColor','k','FaceColor','k')
xlim([0 200])
xlabel('speed (cm/s)','interpreter','latex')
ylabel('Speed Count','interpreter','latex')
title("Speed trial number= "+ selected_trial+" for $PGD\geq0.5$",'interpreter','latex')
%%
figure
% a trial PGD over time
selected_trial= 259;

for i= 1:size(PGD,2)
    if i<241 & PGD(selected_trial,i)>=0.5
        bar(speed(selected_trial,i),'EdgeColor','k','FaceColor','k')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','k')
    elseif i<241 & PGD(selected_trial,i)< 0.5
        bar(speed(selected_trial,i),'EdgeColor','r','FaceColor','r')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','k')
    elseif i>=241 & PGD(selected_trial,i)>=0.5
        bar(speed(selected_trial,i),'EdgeColor','b','FaceColor','b')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','b')
    elseif i>=241 & PGD(selected_trial,i)<0.5
        bar(speed(selected_trial,i),'EdgeColor','r','FaceColor','r')
        title("Trial Number: "+selected_trial+", Time: "+Time(i)+" (s)",'Interpreter','latex','color','b')
    end
    ylim([min(speed(selected_trial,:)) max(speed(selected_trial,:))])
    pause(0.1)
    thisFrame = getframe(gcf);
    myMovie(i) = thisFrame;
end

% %Write the videos
% writerObj = VideoWriter("Speed over time");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);
%% Fitted Circular–Linear Model
% greedy algorithm
clc

time_point= 200; %500
trial_selected= 259;
electrode_phase_instant= rad2deg(squeeze(electrode_phase(:,:,time_point,trial_selected)));
alpha= 0:5:360; sigma= 0:0.5:50;
teta_hat_prim= zeros([size(electrode_phase_instant),length(alpha),length(sigma)]);
r= [];

for k= 1:length(alpha)
    for t= 1:length(sigma)
        a= sigma(t)*cosd(alpha(k));
        b= sigma(t)*sind(alpha(k));
        
        for i= 1:size(electrode_phase_instant,1)
            for j= 1:size(electrode_phase_instant,2)
                teta_hat_prim(i,j,k,t)= -mod(a*((i-1)*0.4)+b*((10-j)*0.4),360);
            end
        end
        r(k,t)= sqrt(nanmean(cosd(squeeze(teta_hat_prim(:,:,k,t)))-electrode_phase_instant,'all').^2+...
            nanmean(sind(squeeze(teta_hat_prim(:,:,k,t)))-electrode_phase_instant,'all').^2);
    end
end



[M,I]= find(r==max(r,[],'all'));
a= sigma(I)*cosd(alpha(M));
b= sigma(I)*sind(alpha(M));

for i= 1:size(electrode_phase_instant,1)
    for j= 1:size(electrode_phase_instant,2)
        teta_hat(i,j)= -mod(a*((10-i)*0.4)+b*((j-1)*0.4),360);
    end
    if teta_hat(i,j)==0
        teta_hat(i,j)=-360;
    end
end

figure
surf(1:5,1:10,teta_hat'+200)
colormap jet

hold on

X= [ones(1,10),repmat(2,1,10),repmat(3,1,10),repmat(4,1,10),repmat(5,1,10)];
Y= repmat(1:10,1,5);
Z= [];
for i=1:5
    Z= [Z,electrode_phase_instant(i,:)];
end

scatter3(X,Y,Z','fill','MarkerFaceColor','k')
zlabel('relative phase (degree)','interpreter','latex')
grid minor
title("Fitted plane in trial number: "+trial_selected+" ,Time: "+time_point,'interpreter','latex')

%% Fitted Circular–Linear Model
clc
% fit plane with regression and alternate PGD
% trial, either

time_point= 200;  %100
trial_selected= 259;
electrode_phase_instant= rad2deg(squeeze(electrode_phase(:,:,time_point,trial_selected)));

X1= [ones(1,10),repmat(2,1,10),repmat(3,1,10),repmat(4,1,10),repmat(5,1,10)];
X2= repmat(1:10,1,5);
y= [];
for i=1:5
    y= [y,electrode_phase_instant(i,:)];
end

X = [ones(size(X1')) X1(:) X2(:)];
b = regress(y',X);

figure
scatter3(X1,X2,y','filled','MarkerFaceColor','k')
hold on

x1fit = min(X1):1:max(X1);
x2fit = min(X2):1:max(X2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;

surf(X1FIT,X2FIT,YFIT)
colormap jet
zlabel('relative phase (degree)','interpreter','latex')
title("Fitted plane in trial number: "+trial_selected+" ,Time: "+time_point,'interpreter','latex')
grid minor


hold on

for i= 1:50
    plot3([X1FIT(i), X1(i)],[X2FIT(i), X2(i)],[YFIT(i), y(i)],'k')
    hold on
end

%% Alternate formulation for PGD
clc
trial_selected= 259;
X1= [ones(1,10),repmat(2,1,10),repmat(3,1,10),repmat(4,1,10),repmat(5,1,10)];
X2= repmat(1:10,1,5);
X = [ones(size(X1')) X1(:) X2(:)];
n= 48; k= 3;

for s=1:size(electrode_phase,4)
    for j=1:size(electrode_phase,3)
        electrode_phase_instant= rad2deg(squeeze(electrode_phase(:,:,j,trial_selected)));
        y= [];
        for i=1:5
            y= [y,electrode_phase_instant(i,:)];
        end
        b = regress(y',X);
        
        x1fit = min(X1):1:max(X1);
        x2fit = min(X2):1:max(X2);
        [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
        YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
        
        YFIT_reshape= [];
        YFIT=YFIT';
        
        for k=1:5
            YFIT_reshape= [YFIT_reshape,YFIT(k,:)];
        end
        
        ro_cc= nansum(sind(y-nanmean(y)).*sind(YFIT_reshape-nanmean(YFIT_reshape)))./sqrt(nansum(sind(y-nanmean(y)).^2).*...
            nansum(sind(YFIT_reshape-nanmean(YFIT_reshape)).^2));
        ro_adj_square(s,j)= 1-((1-ro_cc^2)*(n-1)/(n-k-1));
        
    end
end


figure
histogram(ro_adj_square,'EdgeColor','k','FaceColor','k')
title('Average PGD Over Time','interpreter','latex')
xlabel('PGD','interpreter','latex')
ylabel("$PGD$ count",'interpreter','latex')
%% Plane Wave over times
clc

trial_selected= 259;
X1= [ones(1,10),repmat(2,1,10),repmat(3,1,10),repmat(4,1,10),repmat(5,1,10)];
X2= repmat(1:10,1,5);
X = [ones(size(X1')) X1(:) X2(:)];
n= 48; k= 3; s=0;

figure
for j=1:size(electrode_phase,3)
    if ro_adj_square(trial_selected,j)>=0.5
        s= s+1;
        electrode_phase_instant= rad2deg(squeeze(electrode_phase(:,:,j,trial_selected)));
        y= [];
        for i=1:5
            y= [y,electrode_phase_instant(i,:)];
        end
        b = regress(y',X);
        
        x1fit = min(X1):1:max(X1);
        x2fit = min(X2):1:max(X2);
        [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
        YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
        
        YFIT_reshape= [];
        YFIT=YFIT';
        
        for k=1:5
            YFIT_reshape= [YFIT_reshape,YFIT(k,:)];
        end
        
        %         ro_cc= nansum(sind(y-nanmean(y)).*sind(YFIT_reshape-nanmean(YFIT_reshape)))./sqrt(nansum(sind(y-nanmean(y)).^2).*...
        %             nansum(sind(YFIT_reshape-nanmean(YFIT_reshape)).^2));
        %         ro_adj_square= 1-((1-ro_cc^2)*(n-1)/(n-k-1));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        scatter3(X1,X2,y','filled','MarkerFaceColor','k')
        hold on
        surf(X1FIT,X2FIT,YFIT')
        colormap jet
        colorbar
        caxis([-360 360])
        zlim([-360 360])
        zlabel('relative phase (degree)','interpreter','latex')
        grid minor
        hold on
        for i= 1:50
            plot3([X1FIT(i), X1(i)],[X2FIT(i), X2(i)],[YFIT(i), y(i)],'k')
            hold on
            
        end
        title("Fitted plane in trial number: "+trial_selected+" ,Time: "+Time(j)+" ,PGD= "+ro_adj_square(trial_selected,j),...
            'interpreter','latex')
        drawnow()
        pause(0.1)
        hold off
        thisFrame = getframe(gcf);
        myMovie(s) = thisFrame;
    end
end

% %Write the videos
% writerObj = VideoWriter("fitted Plane Wave over time");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);

%% Cos (phase) During Different Time Points with PGD & speed
clc
select_trial= 100; % 259,50

figure;
colormap hot
for i= 1:size(electrode_phase,3)
    cla reset;
    %imagesc(1:5,1:10,cos(squeeze(electrode_phase(:,:,i,select_trial))));
    pcolor(cos(squeeze(electrode_phase(:,:,i,select_trial))))
    shading interp
    title("Time: "+Time(i)+" s"+", PGD: "+PGD(select_trial,i)+", speed"+speed(select_trial,i),'interpreter','latex')
    drawnow
    thisFrame = getframe(gcf);
    myMovie(i) = thisFrame;
end

% %Write the videos
% writerObj = VideoWriter("wave_shading with info");
% writerObj.FrameRate = 20;
% open(writerObj);
% numberOfFrames = length(myMovie);
% for frameNumber = 1 : numberOfFrames
%     writeVideo(writerObj, myMovie(frameNumber));
% end
% close(writerObj);

%% Preferred direction
clc
velocity_direction= [];
for j= 1:size(electrode_phase,4) % for each trial
    for i= 1:size(electrode_phase,3) % for each time
        instantaneous_phase= squeeze(electrode_phase(:,:,i,j));
        [px(i,j,:,:),py(i,j,:,:)]= gradient(instantaneous_phase);
        velocity_direction(j,i)= nanmean(squeeze(px(i,j,:,:)+1i*py(i,j,:,:)),'all');
    end
end
dir= atan2(imag(velocity_direction),real(velocity_direction));

figure
polarhistogram(-dir,50,'facecolor','k','edgecolor','k')
title('Preferrdd Direction All trials and Times','interpreter','latex')

figure
histogram(mean(dir,1))
xlabel('Preferred Direction (rad)','interpreter','latex')
ylabel('Count','interpreter','latex')

%% Check dirction correctness (Shuffling)
clc
electrode_phase_shuffel= zeros(size(electrode_phase));

clc
for i= 1:size(electrode_phase,1)
    for j= 1:size(electrode_phase,2)
        if ~isnan(electrode_phase(i,j))
        for k= 1:size(electrode_phase,4)
          time_num= randi(size(electrode_phase,3),1,size(electrode_phase,3));  
          for tr= 1:size(electrode_phase,3)
          electrode_phase_shuffel(i,j,tr,k)= electrode_phase(i,j,time_num(t),k);
          end  
        end
        else
           electrode_phase_shuffel(i,j,:,:)= NaN; 
        end
    end
end


px=[]; py=[];

for j= 1:size(electrode_phase_shuffel,4) % for each trial
    for i= 1:size(electrode_phase_shuffel,3) % for each time
        instantaneous_phase= squeeze(electrode_phase_shuffel(:,:,i,j));
        [px(i,j,:,:),py(i,j,:,:)]= gradient(instantaneous_phase);
        velocity_direction(j,i)= nanmean(squeeze(px(i,j,:,:)+1i*py(i,j,:,:)),'all');
    end
end

%%%%%%%%%%%%%%%%% Perefered direction %%%%%%%%%%%%%%%%%
clc
dir= atan2(imag(velocity_direction),real(velocity_direction));

figure
polarhistogram(-dir,'facecolor','k','edgecolor','k')
title('Preferrdd Direction All trials and Times','interpreter','latex')

%% Wave speed histogram
clc
figure
histogram(speed,'facecolor','k','edgecolor','k')
xlim([0 200])
xlabel("Speed cm/s",'interpreter','latex')
ylabel('count speed','interpreter','latex')

%% extra work (PGD as a function of frequency)
clc
bandwidth= 10; s=0; PGD= [];
LFP_signal_filtered= zeros([length(chan),size(chan(1).lfp)]);

for fc= 11:10:111
    s= s+1;
    for i= 1:size(ChannelPosition,1)
        for j= 1:size(ChannelPosition,2)
            if ~isnan(ChannelPosition(i,j))
                [b,a] = butter(order,[fc-bandwidth fc+bandwidth]./fs,'bandpass');
                LFP_signal_filtered(ChannelPosition(i,j),:,:) = filtfilt(b,a,squeeze(LFP_signal(ChannelPosition(i,j),:,:)));
            end
        end
    end
    
    sig_angle= zeros(size(LFP_signal_filtered));
    
    for i= 1:size(LFP_signal_filtered,1)
        sig_angle(i,:,:)= angle(hilbert(squeeze(LFP_signal_filtered(i,:,:))));
    end
    
    electrode_phase= zeros([size(ChannelPosition), size(sig_angle,[2,3])]);
    
    for i= 1:size(ChannelPosition,1)
        for j= 1:size(ChannelPosition,2)
            if ~isnan(ChannelPosition(i,j))
                electrode_phase(i,j,:,:)= squeeze(sig_angle(ChannelPosition(i,j),:,:));
            else
                electrode_phase(i,j,:,:)= NaN;
            end
        end
    end
    
    px=[]; py=[];
    
    for j= 1:size(electrode_phase,4) % for each trial
        for i= 1:size(electrode_phase,3) % for each time
            instantaneous_phase= squeeze(electrode_phase(:,:,i,j));
            [px(i,j,:,:),py(i,j,:,:)]= gradient(instantaneous_phase);
            PGD(s,j,i)= norm(nanmean(squeeze(px(i,j,:,:)),'all')+1i.*nanmean(squeeze(py(i,j,:,:)),'all'))./...
                nanmean(sqrt(squeeze(px(i,j,:,:)).^2+squeeze(py(i,j,:,:)).^2),'all');
        end
    end
end


figure
plot(1:10:110,mean(mean(PGD,2),3),'k')
hold on
xline(10,'--k')
xline(45,'--k')
xlabel('Frequency (Hz)','interpreter','latex')
ylabel('PGD','interpreter','latex')
ylim([ min(mean(mean(PGD,2),3)) 0.45])
%% PPL
clc
N= size(electrode_phase,4);
PPL= zeros(size(electrode_phase,1),size(electrode_phase,2),size(electrode_phase,3));
H= zeros(size(electrode_phase,1),size(electrode_phase,2),size(electrode_phase,3));
Hmax= log2(N);

for x= 1:size(electrode_phase,1)
    for y= 1:size(electrode_phase,2)
        if ~isnan(ChannelPosition(x,y))
            for t= 1:size(electrode_phase,3)
                Pk= electrode_phase(x,y,t,:);
                h= histogram(Pk,5);
                Edges = h.BinEdges;
                diff_Efges= diff(Edges);
                for k= 1:length(diff_Efges)
                    Pk= abs(electrode_phase(x,y,t,k)/diff_Efges(k));
                    H(x,y,t)=  H(x,y,t) - (Pk*log2(Pk));
                end
                PPL(x,y,t)= 100*(1-H(x,y,t)/Hmax);
            end
        else
            PPL(x,y,:)= NaN;
        end
    end
end


PPL_reshape= zeros(size(electrode_phase,1)*size(electrode_phase,2),size(electrode_phase,3));

for i= 1:size(electrode_phase,3)
    PPL_reshape(:,i)=  reshape(PPL(:,:,i)',1,[]);
end
PPL_reshape(1,:)= []; PPL_reshape(40,:)= []; 


figure
imagesc(Time,1:48,sort(PPL_reshape,1,'descend'))
colormap jet
xlabel('Time (s)','interpreter','latex')
set(gca,'YTickLabel',[])
xline(0,'--k','linewidth',7)
