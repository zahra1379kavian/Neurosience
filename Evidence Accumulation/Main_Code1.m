clc
close all
clear

% part 1

%%% Initial Parameters
bias= 1; dt= 0.01; mu= 0; time_interval= 0.1; sigma= 1;
%%% Simple Model
[X, Choice]= simple_model(bias, sigma, mu, dt, time_interval);

%%
clc
% Effect of Brownian Motion Term Variance
bias= 1; mu= 0; sigma= 1;
dt= 0.001:0.05:2; %%% different motion variance
num_iteration= 30;
time_interval= num_iteration.*dt;
X= zeros(length(time_interval),num_iteration);

for i= 1:length(dt)
    [X(i,:), Choice]= simple_model(bias, sigma, mu, dt(i), time_interval(i));
    X(i,:)= X(i,:)/max(X(i,:));
end

figure
imagesc(imgaussfilt(X))
colormap jet
col= colorbar;
xlabel('Iteration','interpreter','latex')
ylabel('Variance of Brownian Motion Term','interpreter','latex')
h= get(col,'Title');
set(h,'String','Evidence Value','interpreter','latex')
title('Effect of Brownian Motion Variance on the Decision Variable','interpreter','latex');
set(gca,'YDir','normal')
yticklabels([0.2,0.45,0.7,0.95,1.20,1.45,3.7,1.95])

%%
% part 2-1

dt= 0.1; mu= 0; time_interval= 10; sigma= 1; bias= 1;
num_trial= 30000; n_bin= 20;
X= [];

for i= 1:num_trial
    [X(i,:), Choice(i)]= simple_model(bias, sigma, mu, dt, time_interval);
    evidence_value(i)= X(i,end);
end

figure
histogram(evidence_value,n_bin,'Normalization','pdf','EdgeColor','k','FaceColor','k')
xlabel('Decsion Value','interpreter','latex')
ylabel('Count of Evidence Value','interpreter','latex')
title("The distribution of Evidence Variable Over "+num2str(num_trial)+ " trials",'interpreter','latex')

figure
histogram(Choice)
title('Result distribution','interpreter','latex')
%%
% part 2-2

% Bias Effect on the Decision Variable
dt= 0.1; mu= 0; time_interval= 10; sigma= 1;
t= 0:dt:time_interval-dt;
bias= [-1,0,0.1,1,10];
figure
for i= 1:length(bias)
    [X, Choice]= simple_model(bias(i), sigma, mu, dt, time_interval);
    plot(t,X,'linewidth',2)
    hold on
end

xlim([0 time_interval])
title("Bias Effect on the  Decision Variable",'interpreter','latex')
legend('bias = -1','bias = 0','bias = 0.1','bias = 1','bias = 10',...
    'location','best','interpreter','latex')
xlabel('Time (s)','interpreter','latex')
ylabel('Decsion Value','interpreter','latex')
grid minor
%%
% part 3

dt= 0.1; mu= 0;  sigma= 1; bias= 0.1;
time_interval= 0.5:dt:10;
num_trial= 20000;
evidence_value= zeros(1,length(time_interval));

for j= 1:length(time_interval)
    for i= 1:num_trial
        [~, Choice]= simple_model(bias, sigma, mu, dt, time_interval(j));
        evidence_value(j)= evidence_value(j)+ (1-Choice);
    end
end

figure
plot(time_interval,evidence_value./num_trial*100,'linewidth',2,'color','k')
xlabel('Time Interval (s)','interpreter','latex')
ylabel('Error Rate','interpreter','latex')
title('Error Rate Variation over Time','interpreter','latex')
grid minor
%%
clc
% part 4

dt= 0.1; mu= 0;  sigma= 1; bias= 0.1; X=[];
time_interval= 10;
num_trial= 10000;
t= 0:dt:time_interval-dt;

for i= 1:num_trial
    [X(i,:), ~]= simple_model(bias, sigma, mu, dt, time_interval);
end

figure
for i= 1:num_trial
    plot(t,(X(i,:)),'--k')
    hold on
end
xlabel('Time (s)','interpreter','latex')
ylabel('Decsion Value','interpreter','latex')
title("Trajectories Over Time in "+num2str(num_trial)+ " Trials",'interpreter','latex')
grid minor

figure
subplot(1,2,1)
plot(t,mean(X),'LineWidth',1.5,'color','k');
xlabel('Time (s)','interpreter','latex')
ylabel('Mean of Decsion Value','interpreter','latex')
title("Average of Decision variable Mean Over Time in "+num2str(num_trial)+ " Trials",'interpreter','latex')
grid minor

subplot(1,2,2)
plot(t,mean(X)+std(X),'LineWidth',1.5,'color','r')
hold on
plot(t,mean(X)-std(X),'LineWidth',1.5,'color','g')
xlabel('Time (s)','interpreter','latex')
ylabel('Standard Deviation','interpreter','latex')
title("Standard Deviation of Decision variable Over Time in "+num2str(num_trial)+ " Trials",'interpreter','latex')
legend('above the maen','below the mean','location','best','interpreter','latex')
grid minor
%%
clc
% part 5
bias= 1; sigma= 1; dt= 0.1;
start_point= [bias*dt-0.05 bias*dt bias*dt+0.1];
num_iter= 10000;

for j = 1:length(start_point)
    for i= 1:num_iter
        Choice(j,i)= simple_model2(start_point(j), sigma, bias, dt);
    end
end

figure
subplot(1,3,1)
histogram(Choice(1,:))
title("Start Point: "+start_point(1),'interpreter','latex')
subplot(1,3,2)
histogram(Choice(2,:))
title("Start Point: "+start_point(2),'interpreter','latex')
subplot(1,3,3)
histogram(Choice(3,:))
title("Start Point: "+start_point(3),'interpreter','latex')
sgtitle("Result Distribution, $N(0.1, \sqrt 0.1)$",'interpreter','latex')
%%
clc
% part 6
positive_thr= 10; %4 \ 10 \ 4      %%%% Run this part for these threshold value
negative_thr= -4; %-4 \ -4 \ -10
sigma= 1;
X0= 1;
bias_pos= 0.1;
bias_neg= 0.1;
num_trials= 20000;

RT= zeros(1,num_trials); Choice= zeros(1,num_trials); X= cell(1,num_trials);

for i= 1:num_trials
    [X{i}, RT(i), Choice(i)]= two_choice_trial(positive_thr, negative_thr,sigma, X0, bias_pos,bias_neg);
    
end

RT_correct= RT(Choice==1);
RT_false= RT(Choice==-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
histogram(RT_correct,'Normalization','probability')
hold on
histogram(RT_false,'Normalization','probability')
xlabel('Reaction Time','interpreter','latex')
ylabel('Count of Trials','interpreter','latex')
title('Reaction Time - Free Response Model','interpreter','latex')
legend('Correct Response','Incorrect Response','interpreter','latex')
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(1,2,1)
h= histfit(RT_correct,[],'inversegaussian');
h(1).FaceColor= [0.8 0.8 1];
h(2).Color = [.2 .2 .2];
xlabel('Reaction Time','interpreter','latex')
ylabel('Count of Trials','interpreter','latex')
title('Reaction Time of Correct trials - Free Response Model','interpreter','latex')
grid minor

subplot(1,2,2)
h= histfit(RT_false,[],'inversegaussian');
h(1).FaceColor= [0.8 0.8 1];
h(2).Color = [.2 .2 .2];
xlabel('Reaction Time','interpreter','latex')
title('Reaction Time of Incorrcet Trials - Free Response Model','interpreter','latex')
grid minor

correct_indx= find(Choice==1);
incorrect_indx= find(Choice==-1);

correct_rnd_indx= correct_indx(randi(length(correct_indx),[1,5]));
incorrect_rnd_indx= incorrect_indx(randi(length(incorrect_indx),[1,5]));

figure
for i= 1:5
    x= X{correct_rnd_indx(i)};
    plot(x);
    hold on
end
hold on
yline(positive_thr,'--k','Positive Threshold','interpreter','latex')
yline(negative_thr,'--k','Negative Threshold','interpreter','latex')
ylim([negative_thr-1 positive_thr+1])
grid minor
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Correct Trials','interpreter','latex')



figure
for i= 1:5
    plot(X{incorrect_rnd_indx(i)});
    hold on
end
hold on
yline(positive_thr,'--k','Positive Threshold','interpreter','latex')
yline(negative_thr,'--k','Negative Threshold','interpreter','latex')
ylim([negative_thr-1 positive_thr+1])
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Incorrect Trials','interpreter','latex')
grid minor
%%
clc
% use different bias for positive or negative random motion
positive_thr= 2;
negative_thr= -2; %-1
sigma= 1;
X0= 1;
bias_neg= 0.1;
bias_pos= 1;
num_trials= 20000;

RT= zeros(1,num_trials); Choice= zeros(1,num_trials); X= cell(1,num_trials);

for i= 1:num_trials
    [X{i}, RT(i), Choice(i)]= two_choice_trial(positive_thr, negative_thr,sigma, X0, bias_pos, bias_neg);
    
end

RT_correct= RT(Choice==1);
RT_false= RT(Choice==-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
histogram(RT_correct,'Normalization','probability')
hold on
histogram(RT_false,'Normalization','probability')
xlabel('Reaction Time','interpreter','latex')
ylabel('Count of Trials','interpreter','latex')
title('Reaction Time - Free Response Model','interpreter','latex')
legend('Correct Response','Incorrect Response','interpreter','latex')
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(1,2,1)
h= histfit(RT_correct,[],'inversegaussian');
h(1).FaceColor= [0.8 0.8 1];
h(2).Color = [.2 .2 .2];
xlabel('Reaction Time','interpreter','latex')
ylabel('Count of Trials','interpreter','latex')
title('Reaction Time of Correct trials - Free Response Model','interpreter','latex')
grid minor

subplot(1,2,2)
h= histfit(RT_false,[],'inversegaussian');
h(1).FaceColor= [0.8 0.8 1];
h(2).Color = [.2 .2 .2];
xlabel('Reaction Time','interpreter','latex')
title('Reaction Time of Incorrcet Trials - Free Response Model','interpreter','latex')
grid minor

% plot the decision variable over time and for different bias and start
% point. Give different bias for corrcet and incorrect

correct_indx= find(Choice==1);
incorrect_indx= find(Choice==-1);

correct_rnd_indx= correct_indx(randi(length(correct_indx),[1,5]));
incorrect_rnd_indx= incorrect_indx(randi(length(incorrect_indx),[1,5]));

figure
for i= 1:5
    plot(X{correct_rnd_indx(i)});
    hold on
    % plot(X{incorrect_rnd_indx(i)});
end
hold on
yline(positive_thr,'--k','Positive Threshold')
yline(negative_thr,'--k','Negative Threshold')
ylim([negative_thr-1 positive_thr+1])
grid minor
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Correct Trials','interpreter','latex')



figure
for i= 1:5
    plot(X{incorrect_rnd_indx(i)});
    hold on
end
hold on
yline(positive_thr,'--k','Positive Threshold')
yline(negative_thr,'--k','Negative Threshold')
ylim([negative_thr-1 positive_thr+1])
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Incorrect Trials','interpreter','latex')


%%
% part 7 (extension of drift diffusion model)
clc

positive_thr= 10;  negative_thr= -10;
sigma1= 1;  sigma2= 1; %[1 1], [0.1 1], [1 0.1]
X01= 0;   X02= 0; %[0 0], [5 0], [0 5]
bias1= 1;  bias2= -0.5; %[0.1 -0.1], [0.1 -1], [1 -0.5]
num_trials= 10000;

RT= zeros(1,num_trials); Choice= zeros(1,num_trials); X1= cell(1,num_trials); X2= cell(1,num_trials);


for i= 1:num_trials
    [X1{i}, X2{i}, RT(i), Choice(i)]= race_trial(positive_thr, negative_thr, sigma1, sigma2, X01, X02, bias1, bias2);
end

% RT_correct= RT(Choice==1);
% RT_false= RT(Choice==-1);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% histogram(RT_correct,'Normalization','probability')
% hold on
% histogram(RT_false,'Normalization','probability')
% xlabel('Reaction Time','interpreter','latex')
% ylabel('Count of Trials','interpreter','latex')
% title('Reaction Time - Free Response Model','interpreter','latex')
% legend('Correct Response','Incorrect Response','interpreter','latex')
% grid minor
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% figure
% subplot(1,2,1)
% h= histfit(RT_correct,[],'inversegaussian');
% h(1).FaceColor= [0.8 0.8 1];
% h(2).Color = [.2 .2 .2];
% xlabel('Reaction Time','interpreter','latex')
% ylabel('Count of Trials','interpreter','latex')
% title('Reaction Time of Correct trials - Free Response Model','interpreter','latex')
% grid minor
% 
% subplot(1,2,2)
% h= histfit(RT_false,[],'inversegaussian');
% h(1).FaceColor= [0.8 0.8 1];
% h(2).Color = [.2 .2 .2];
% xlabel('Reaction Time','interpreter','latex')
% title('Reaction Time of Incorrcet Trials - Free Response Model','interpreter','latex')
% grid minor

correct_indx= find(Choice==1);
incorrect_indx= find(Choice==-1);

correct_rnd_indx= correct_indx(randi(length(correct_indx)));
incorrect_rnd_indx= incorrect_indx(randi(length(incorrect_indx)));

figure
stairs(X1{correct_rnd_indx},'linewidth',1.5);
hold on
stairs(X2{correct_rnd_indx},'linewidth',1.5);
hold on
yline(positive_thr,'--k','Positive Threshold')
yline(negative_thr,'--k','Negative Threshold')
ylim([negative_thr-1 positive_thr+1])
grid minor
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Correct Trials','interpreter','latex')
legend('X1','X2','interpreter','latex','location','best')

figure
stairs(X1{incorrect_rnd_indx},'linewidth',1.5);
hold on
stairs(X2{incorrect_rnd_indx},'linewidth',1.5);

hold on
yline(positive_thr,'--k','Positive Threshold')
yline(negative_thr,'--k','Negative Threshold')
ylim([negative_thr-1 positive_thr+1])
grid minor
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Incorrect Trials','interpreter','latex')
legend('X1','X2','interpreter','latex','location','best')


%%
clc
% part 8

positive_thr= 6;  negative_thr= -6;
sigma1= 1;  sigma2= 1;
X01= 2;   X02= -2;
bias1= 0.1;  bias2= -0.1; 
num_trials= 20000;
time_interval= 10;
t= 0:dt:time_interval-dt;

Choice= zeros(1,num_trials); X1= cell(1,num_trials); X2= cell(1,num_trials);


for i= 1:num_trials
    [X1{i}, X2{i}, Choice(i)]= extend_race_trial(positive_thr, negative_thr, sigma1, sigma2, X01, X02, bias1, bias2, time_interval);
end

correct_indx= find(Choice==1);
incorrect_indx= find(Choice==-1);

correct_rnd_indx= correct_indx(randi(length(correct_indx)));
incorrect_rnd_indx= incorrect_indx(randi(length(incorrect_indx)));

figure
stairs(X1{correct_rnd_indx},'linewidth',1.5);
hold on
stairs(X2{correct_rnd_indx},'linewidth',1.5);
hold on
yline(positive_thr,'--k','Positive Threshold')
yline(negative_thr,'--k','Negative Threshold')
ylim([negative_thr-1 positive_thr+1])
grid minor
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Correct Trials','interpreter','latex')
legend('X1','X2','interpreter','latex','location','best')

figure
stairs(X1{incorrect_rnd_indx},'linewidth',1.5);
hold on
stairs(X2{incorrect_rnd_indx},'linewidth',1.5);

hold on
yline(positive_thr,'--k','Positive Threshold')
yline(negative_thr,'--k','Negative Threshold')
ylim([negative_thr-1 positive_thr+1])
grid minor
xlabel('Time (s)','interpreter','latex')
ylabel('Descision Value','interpreter','latex')
title('Incorrect Trials','interpreter','latex')
legend('X1','X2','interpreter','latex','location','best')

