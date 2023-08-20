function [DS_trials, state, TA_bad_fractal, TP_bad_fractal, TP_good_fractal, good_label, bad_label, TA_trials, 
TP_trials]= Randomization(num_trial,DS_opt)

% randomly assign DS
DS_trials= zeros(1,num_trial);

for i=1:4:num_trial
   DS_trials(i:i+3)= DS_opt(randperm(4,4)); %%%%%%%%%%%%%
end


% randomly label images
[good_label, bad_label]= label_func();%%%%%%

% randomly assign TP, TA
TA_trials= randperm(num_trial,num_trial/2);
trials= zeros(1,num_trial);
trials(TA_trials)= 1;
TP_trials= find(trials==0);

state= zeros(1,num_trial);
state(TP_trials)=1; %%%%%%%%%%%

% randomly assign images for TA
TA_trials_image= zeros(1,sum(DS_trials(TA_trials)));
L= length(TA_trials_image); f= floor(L/24);
TA_trials_image(1:f*24)= repmat(1:24,1,f);
TA_trials_image(f*24+1:end)= randperm(24,L-f*24);
TA_bad_fractal= bad_label(TA_trials_image); %%%%%%%%%%



% randomly assign images for TP
% bad
TP_trials_image= zeros(1,sum(DS_trials(TP_trials)-1));
L= length(TP_trials_image); f= floor(L/24);
TP_trials_image(1:f*24)= repmat(1:24,1,f);
TP_trials_image(f*24+1:end)= randperm(24,L-f*24);
TP_bad_fractal= bad_label(TP_trials_image); %%%%%%%%%%


% good
TP_trials_image= zeros(1,length(TP_trials));
L= length(TP_trials_image); f= floor(L/24);
TP_trials_image(1:f*24)= repmat(1:24,1,f);
TP_trials_image(f*24+1:end)= randperm(24,L-f*24);
TP_good_fractal= good_label(TP_trials_image); %%%%%%%%%%
end
