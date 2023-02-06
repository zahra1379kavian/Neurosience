close all
clc
clear

%%%loading data
fs = 200;
data = struct2cell(load('Stimulus'));
stimulus = data{1, 1}.Stimulus;
spike_time = data{1, 1}.Spike_Times;


%%
%%%Stimulus and spike-time 50:60(s)
f = 50:1/fs:60;

x = zeros(1,10*200+1);

figure
plot(f,stimulus(50:50+10*fs),'Color','black');
hold on
for i = 1:size(spike_time,2)
    if spike_time(1,i)>=50 & spike_time(1,i)<=60
       %indx = (spike_time(1,i)-50)/(1/200); 
       %x(1,indx) = 1;
       x = [spike_time(1,i) spike_time(1,i)];
       y = [100 90];
       line(x,y,'Color','red');
    end 
end
xlabel('Time(s)','interpreter','latex')
ylabel('Sound Level(dB)','interpreter','latex')

%%
%%%stimulus_ensemble matrix
time_window = 0.2;
s = 1;

for i = 1:size(stimulus,2)-time_window*fs
    stimulus_ensemble(s,:) = stimulus(1,i:i+time_window*fs);
    s = s+1;
end

figure
imagesc(stimulus_ensemble)
colormap('hot')
%%
%%%SpikeTrigger_ensemble matrix

s = 1;
for i = 1:size(spike_time,2)
    indx1 = spike_time(1,i)*fs-0.1*fs;
    indx2 = spike_time(1,i)*fs+0.1*fs;
    A = stimulus(1,indx1:indx2);
    SpikeTrigger_ensemble(s,:) = stimulus(1,indx1:indx2);
    s = s+1;
    
end

figure
imagesc(SpikeTrigger_ensemble)
colormap('hot')
%%
%%%Calculate STA

STA = mean(SpikeTrigger_ensemble,1);
t = 0:1/fs:0.2;

figure
plot(t,STA,'linewidth',2)
xlabel('Time(s)','interpreter','latex')




