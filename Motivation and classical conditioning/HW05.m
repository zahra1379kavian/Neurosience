close all
clc
clear

%% Pavlovian
% initial parameters
epsi= 0.05; reward= 1; trial_number= 200; w= zeros(size(1:trial_number)); u=0; value= w(1)*u;
u= 1;
for i= 2:trial_number
    error= reward-value;
    w(i)= w(i-1)+epsi*u*error;
    value= w(i)*u;
end
figure
scatter(1:trial_number,w,'linewidth',2,'MarkerEdgeColor','k')
xlabel('trial number','interpreter','latex')
ylabel('$w$','interpreter','latex')
title('Pavlovian Paradigm Conditioning - RW Rule','interpreter','latex')

%% Extinction paradigm

epsi= 0.05; reward= 1; trial_number= 200; w= zeros(size(1:trial_number)); u=0; value= w(1)*u;
u= 1;

for i= 2:trial_number
    if i<=trial_number/2
        reward= 1;
        error= reward-value;
        w(i)= w(i-1)+epsi*u*error;
        value= w(i)*u;
    else
        reward= 0;
        error= reward-value;
        w(i)= w(i-1)+epsi*u*error;
        value= w(i)*u;
    end
    
end

figure
scatter(1:trial_number,w,'linewidth',2,'MarkerEdgeColor','k')
hold on
xline(trial_number/2,'--k','linewidth',1)
xlabel('trial number','interpreter','latex')
ylabel('$w$','interpreter','latex')
title("Pavlovian \& Extinction Paradigm Conditioning - RW Rule",'interpreter','latex')
text(40,1,'Pre-Train','Color','k','FontSize',14,'interpreter','latex')
text(trial_number-40,1,'Train','Color','k','FontSize',14,'interpreter','latex')

%% Partial paradigm
alpha= 50;
reward_trial= randi([2 trial_number],1,trial_number*alpha/100);
epsi= 0.05; trial_number= 200; w= zeros(size(1:trial_number)); u=0; value= w(1)*u;
u= 1;

for i= 2:trial_number
    if ~nnz(ismember(reward_trial,i))
        reward= 1;
        error= reward-value;
        w(i)= w(i-1)+epsi*u*error;
        value= w(i)*u;
    else
        reward= 0;
        error= reward-value;
        w(i)= w(i-1)+epsi*u*error;
        value= w(i)*u;
    end
    
end

figure
scatter(1:trial_number,w,'square','linewidth',2,'MarkerEdgeColor','k')
xlabel('trial number','interpreter','latex')
ylabel('$w$','interpreter','latex')
title(["Partial Paradigm Conditioning "+"stimulus randomly on "+int2str(alpha)+ "\% of trials"+"- RW Rule"],'interpreter','latex')
text(40,1,'Pre-Train','Color','b','FontSize',14,'interpreter','latex')

%% Blocking Paradigm
clc
epsi= 0.05; reward= 1; trial_number= 200; w1= zeros(size(1:trial_number)); w2= zeros(size(1:trial_number/2));
u=0; value1= w1(1)*u; value2= 0;
u1= 1; u2= 1;

for i= 2:trial_number
    if i<=trial_number/2
        error= reward-value1;
        w1(i)= w1(i-1)+epsi*u1*error;
        value1= w1(i)*u1;
    else
        error= reward-value1-value2;
        w1(i)= w1(i-1)+epsi*u1*error;
        w2(i-trial_number/2+1)= w2(i-trial_number/2)+epsi*u2*error;
        value1= w1(i)*u1; value2= w2(i-trial_number/2+1)*u2;
    end
end

figure
plot(1:trial_number,w1,'*-','linewidth',2,'Color','k')
hold on
plot(trial_number/2:trial_number,w2,'--k','linewidth',2)
hold on
xline(trial_number/2,'--k','linewidth',0.5)

text(40,0.5,'Pre-Train','Color','k','FontSize',14,'interpreter','latex')
text(trial_number-40,0.5,'Train','Color','k','FontSize',14,'interpreter','latex')
xlabel('trial number','interpreter','latex')
ylabel('$w$','interpreter','latex')
title("Blocking Paradigm Conditioning - RW Rule",'interpreter','latex')
legend('Stimulus 1','Stimulus 2','Location','best','interpreter','latex')

%% Inhibitory paradigm
clc
epsi= 0.05; reward1= 1; reward2= -1; trial_number= 200; w1= zeros(size(1:trial_number));
w2= zeros(size(1:trial_number)); u=0; value1= w1(1)*u; value2= w2(1)*u;
u1= 1; u2= 1;

for i= 2:trial_number
    error= reward1-value1;
    w1(i)= w1(i-1)+epsi*u1*error;
    value1= w1(i)*u1;
    
    error= reward2-value2;
    w2(i)= w2(i-1)+epsi*u2*error;
    value2= w2(i)*u1;
end


figure
plot(1:trial_number,w1,'-*','linewidth',2,'Color','k')
hold on
plot(1:trial_number,w2,'-*','linewidth',2,'color',[195 195 195]./255)

xlabel('trial number','interpreter','latex')
ylabel('$w$','interpreter','latex')
title("Inhibitory Paradigm Conditioning - RW Rule",'interpreter','latex')

legend('Stimulus 1','Stimulus 2','Location','best','interpreter','latex')

%% Overshadow paradigm
clc
epsi1= 0.1; % 0.05 
epsi2= 0.1; reward= 1; trial_number= 200; w1= zeros(size(1:trial_number)); w2= zeros(size(1:trial_number));
u=0; value1= w1(1)*u; value2= 0;
u1= 1; u2= 1;

for i= 2:trial_number
    error= reward-value1-value2;
    w1(i)= w1(i-1)+epsi1*u1*error; w2(i)= w2(i-1)+epsi2*u2*error;
    value1= w1(i)*u1; value2= w2(i)*u1;
end

figure
plot(1:trial_number,w1,'*-','linewidth',2,'Color','k')
hold on
plot(1:trial_number,w2,'*-','linewidth',2,'Color','b')

xlabel('trial number','interpreter','latex')
ylabel('$w$','interpreter','latex')
title("Overshadow Paradigm Conditioning - RW Rule",'interpreter','latex')
legend("Stimulus 1, learning\_rate: "+epsi1,"Stimulus 2, learning\_rate: "+epsi2,'Location','best','interpreter','latex')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kalman filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% blocking
clc
trial_number= 20; reward= 1; u1=1; u2=0; sigma1= 0.6; sigma2= 0.6; sigma12= 0; v= 0.1; tau= 0.4;
Sigma= zeros(2,2,trial_number); Sigma(:,:,1)= [sigma1,sigma12;sigma12,sigma2];
W= zeros(2,trial_number); W(:,1)= [0;0];

for i= 2:trial_number
    if i== trial_number/2+1, u2=1; end
    if i<= trial_number/2
        delta_Sigma= Sigma(:,:,i-1)*[1;0]*inv([1,0]*Sigma(:,:,i-1)*[1;0]+tau^2)*[1,0]*Sigma(:,:,i-1);
    else
        delta_Sigma= Sigma(:,:,i-1)*[1;1]*inv([1,1]*Sigma(:,:,i-1)*[1;1]+tau^2)*[1,1]*Sigma(:,:,i-1);
    end
    Sigma(:,:,i)= Sigma(:,:,i-1)-delta_Sigma;
    delta_W= (reward-u1*W(1,i-1)-u2*W(2,i-1)).*(Sigma(:,:,i-1)*[u1;u2]./([u1,u2]*Sigma(:,:,i-1)*[u1;u2]+tau^2));
    W(:,i)=  W(:,i-1)+delta_W;
end

figure
subplot(2,1,1)
plot(1:trial_number,W(1,:),'linewidth',2,'Color','k')
hold on
plot(trial_number/2:trial_number,W(2,trial_number/2:end),'--','linewidth',2,'Color','k')
hold on
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
ylim([0 1.3])
xlim([1 trial_number])
ylabel('w(t)','interpreter','latex')
title('Mean','interpreter','latex','position',[18 1.31 0])
text(5,1.1,'$\hat W_{1}(t)$','Color','k','FontSize',14,'interpreter','latex')
text(15,0.3,'$\hat W_{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

subplot(2,1,2)
plot(1:trial_number,squeeze(Sigma(1,1,:)),'linewidth',2,'Color','k')
hold on
plot(trial_number/2:trial_number,squeeze(Sigma(2,2,trial_number/2:end)),'--','linewidth',2,'Color','k')
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
ylim([0 1])
xlim([1 trial_number])
ylabel('$\sigma^{2}(t)$','interpreter','latex')
title('Variance','interpreter','latex','position',[18 1.01 0])
text(5,0.3,'$\sigma_{1}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')
text(15,0.3,'$\sigma_{2}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

sgtitle('Blocking','interpreter','latex','FontSize',20)
%% unBlocking
clc
trial_number= 20; reward= 1; u1=1; u2=0; sigma1= 0.6; sigma2= 0.6; sigma12= 0; v= 0.1; tau= 0.4;
Sigma= zeros(2,2,trial_number); Sigma(:,:,1)= [sigma1,sigma12;sigma12,sigma2];
W= zeros(2,trial_number); W(:,1)= [0;0];

for i= 2:trial_number
    if i== trial_number/2+1, u2=1; reward=2; end
    if i<= trial_number/2
        delta_Sigma= Sigma(:,:,i-1)*[1;0]*inv([1,0]*Sigma(:,:,i-1)*[1;0]+tau^2)*[1,0]*Sigma(:,:,i-1);
    else
        delta_Sigma= Sigma(:,:,i-1)*[1;1]*inv([1,1]*Sigma(:,:,i-1)*[1;1]+tau^2)*[1,1]*Sigma(:,:,i-1);
    end
    Sigma(:,:,i)= Sigma(:,:,i-1)-delta_Sigma;
    delta_W= (reward-u1*W(1,i-1)-u2*W(2,i-1)).*(Sigma(:,:,i-1)*[u1;u2]./([u1,u2]*Sigma(:,:,i-1)*[u1;u2]+tau^2));
    W(:,i)=  W(:,i-1)+delta_W;
end

figure
subplot(2,1,1)
plot(1:trial_number,W(1,:),'linewidth',2,'Color','k')
hold on
plot(trial_number/2:trial_number,W(2,trial_number/2:end),'--','linewidth',2,'Color','k')
hold on
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
ylim([0 1.3])
xlim([1 trial_number])
ylabel('w(t)','interpreter','latex')
title('Mean','interpreter','latex','position',[18 1.31 0])
text(5,1.1,'$\hat W_{1}(t)$','Color','k','FontSize',14,'interpreter','latex')
text(13,0.7,'$\hat W_{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

subplot(2,1,2)
plot(1:trial_number,squeeze(Sigma(1,1,:)),'linewidth',2,'Color','k')
hold on
plot(trial_number/2:trial_number,squeeze(Sigma(2,2,trial_number/2:end)),'--','linewidth',2,'Color','k')
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
ylim([0 1])
xlim([1 trial_number])
ylabel('$\sigma^{2}(t)$','interpreter','latex')
title('Variance','interpreter','latex','position',[18 1.01 0])
text(5,0.3,'$\sigma_{1}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')
text(15,0.3,'$\sigma_{2}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

sgtitle('Unblocking','interpreter','latex','FontSize',20)

%% Backward Blocking
clc

trial_number= 100; % 20
reward= 1; u1=1; u2=1; sigma1= 0.6; sigma2= 0.6; sigma12= 0; v= 0.1; tau= 0.4; 
Sigma= zeros(2,2,trial_number); Sigma(:,:,1)= [sigma1,sigma12;sigma12,sigma2];
W= zeros(2,trial_number); W(:,1)= [0;0];
process_mean= 0; process_var= 1; % 0 0.1 1
measure_mean= 0; measure_var= 0.1; %0 0.1 0.5 1

for i= 2:trial_number
    if i== trial_number/2+1; u2=0; end
    if i> trial_number/2
        %Sigma(:,:,i-1)= Sigma(:,:,i-1)+random('Normal',process_mean,process_var);
        Sigma(:,:,i-1)= Sigma(:,:,i-1)+process_var^2;
        delta_Sigma= Sigma(:,:,i-1)*[1;0]*inv([1,0]*Sigma(:,:,i-1)*[1;0]+tau^2)*[1,0]*Sigma(:,:,i-1);
    else
        %Sigma(1,1,i-1)= Sigma(1,1,i-1)+random('Normal',process_mean,process_var);
        Sigma(1,1,i-1)= Sigma(1,1,i-1)+process_var^2;
        delta_Sigma= Sigma(:,:,i-1)*[1;1]*inv([1,1]*Sigma(:,:,i-1)*[1;1]+tau^2)*[1,1]*Sigma(:,:,i-1);
    end
    Sigma(:,:,i)= Sigma(:,:,i-1)-delta_Sigma;
    G(:,i)= Sigma(:,:,i-1)*[1;0]*inv([1,0]*Sigma(:,:,i-1)*[1;0]+tau^2);
    %delta_W= (reward+random('Normal',measure_mean,measure_var)-u1*W(1,i-1)-u2*W(2,i-1)).*(Sigma(:,:,i-1)*[u1;u2]./([u1,u2]*Sigma(:,:,i-1)*[u1;u2]+tau^2));
    delta_W= (reward+measure_var^2-u1*W(1,i-1)-...
        u2*W(2,i-1)).*(Sigma(:,:,i-1)*[u1;u2]./([u1,u2]*Sigma(:,:,i-1)*[u1;u2]+tau^2));
    W(:,i)=  W(:,i-1)+delta_W;
end

figure
subplot(2,1,1)
plot(1:trial_number,W(1,:),'linewidth',2,'Color','k')
hold on
plot(1:trial_number,W(2,1:end),'--','linewidth',2,'Color','k')
hold on
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
%ylim([0 1.3])
xlim([1 trial_number])
ylabel('w(t)','interpreter','latex')
%title('Mean','interpreter','latex','position',[16 1.31 0])
%text(5,1.1,'$\hat W_{1}(t)$','Color','k','FontSize',14,'interpreter','latex')
%text(13,0.3,'$\hat W_{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

subplot(2,1,2)
plot(1:trial_number,squeeze(Sigma(1,1,:)),'linewidth',2,'Color','k')
hold on
plot(1:trial_number,squeeze(Sigma(2,2,1:end)),'--','linewidth',2,'Color','k')
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
%ylim([0 1])
xlim([1 trial_number])
ylabel('$\sigma^{2}(t)$','interpreter','latex')
%title('Variance','interpreter','latex','position',[16 1.01 0])
%text(5,0.5,'$\sigma_{1}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')
%text(15,0.2,'$\sigma_{2}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

sgtitle("Backward Blocking, Sigma\_Process Noise: "+process_var+...
    ", Sigma\_Measurement Noise: "+measure_var,'interpreter','latex','FontSize',15) %20
%%
% Kalman Filter Gain vs uncertainty matrix
clc
figure
subplot(2,1,1)
scatter(G(1,2:end),squeeze(Sigma(1,1,2:end)),'MarkerEdgeColor','k')
ylabel('Uncertainty Variance','interpreter','latex')
subplot(2,1,2)
scatter(G(2,2:end),squeeze(Sigma(2,2,2:end)),'MarkerEdgeColor','k')
xlabel('Filter Gain','interpreter','latex')
ylabel('Uncertainty Variance','interpreter','latex')

%% Joint distribution in Backward Blocking
clc
trial= 1; mu= W(:,trial)'; sigma= Sigma(:,:,trial);

x1= -1:0.1:2; x2= -1:0.1:2;
[X1,X2]= meshgrid(x1, x2); X= [X1(:) X2(:)];

y= mvnpdf(X,mu,sigma); y= reshape(y,length(x2),length(x1));
level= -1: (max(y)-min(y))/5: 2;

figure
subplot(1,3,1)
contourf(x1,x2,y,level,'edgecolor','none')
colormap gray
hold on
plot(mu(1),mu(2),'k*','MarkerSize',10)
xlabel('$\bar w_{1}$','interpreter','latex','fontsize',14)
ylabel('$\bar w_{2}$','interpreter','latex','fontsize',14)
title('t = 1','interpreter','latex','fontsize',14)

trial= 9; mu= W(:,trial)'; sigma= Sigma(:,:,trial);
level= -1:0.1: 2;
y= mvnpdf(X,mu,sigma); y= reshape(y,length(x2),length(x1));
subplot(1,3,2)
contourf(x1,x2,y,level,'edgecolor','none')
colormap gray
hold on
plot(mu(1),mu(2),'k*','MarkerSize',10)
title('t = 9','interpreter','latex','fontsize',14)


trial= 19; mu= W(:,trial)'; sigma= Sigma(:,:,trial);
level= -5:0.1: 2;
y= mvnpdf(X,mu,sigma); y= reshape(y,length(x2),length(x1));
subplot(1,3,3)
contourf(x1,x2,y,level,'edgecolor','none')
colormap gray
hold on
plot(mu(1),mu(2),'k*','MarkerSize',10)
title('t = 19','interpreter','latex','fontsize',14)
%% reward and punishment
clc
trial_number= 20; reward= 1; u1=1; u2=0; sigma1= 0.6; sigma2= 0.6; sigma12= 0; v= 0.1; tau= 0.4;
Sigma= zeros(2,2,trial_number); Sigma(:,:,1)= [sigma1,sigma12;sigma12,sigma2];
W= zeros(2,trial_number); W(:,1)= [0;0];

for i= 2:trial_number
    if i== trial_number/2+1, reward= -1; end
    if i<= trial_number/2
        delta_Sigma= Sigma(:,:,i-1)*[1;0]*inv([1,0]*Sigma(:,:,i-1)*[1;0]+tau^2)*[1,0]*Sigma(:,:,i-1);
    else
        delta_Sigma= Sigma(:,:,i-1)*[1;1]*inv([1,1]*Sigma(:,:,i-1)*[1;1]+tau^2)*[1,1]*Sigma(:,:,i-1);
    end
    Sigma(:,:,i)= Sigma(:,:,i-1)-delta_Sigma;
    delta_W= (reward-u1*W(1,i-1)-u2*W(2,i-1)).*(Sigma(:,:,i-1)*[u1;u2]./([u1,u2]*Sigma(:,:,i-1)*[u1;u2]+tau^2));
    W(:,i)=  W(:,i-1)+delta_W;
end

figure
subplot(2,1,1)
plot(1:trial_number,W(1,:),'linewidth',2,'Color','k')
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
ylim([0 1.3])
xlim([1 trial_number])
ylabel('w(t)','interpreter','latex')
title('Mean','interpreter','latex','position',[18 1.31 0])
text(5,1.1,'$\hat W_{1}(t)$','Color','k','FontSize',14,'interpreter','latex')

subplot(2,1,2)
plot(1:trial_number,squeeze(Sigma(1,1,:)),'linewidth',2,'Color','k')
xline(trial_number/2,'--','color',[192 192 192]./255,'linewidth',1)
ylim([0 1])
xlim([1 trial_number])
ylabel('$\sigma^{2}(t)$','interpreter','latex')
title('Variance','interpreter','latex','position',[18 1.01 0])
text(5,0.5,'$\sigma_{1}^{2}(t)$','Color','k','FontSize',14,'interpreter','latex')

sgtitle('Reward \& Punishment','interpreter','latex','FontSize',20)

%% an adaptive factor analysis model
clc
close all

num_trial= 100;
sigma_w= 0.1; sigma_r= 0.6; sigma_mu= 0.001;  u= 1;
r= zeros(1,num_trial); w= zeros(1,num_trial);
delta_r= random('Normal',0,sigma_r,size(r)); delta_w= random('Normal',0,sigma_w,size(w));
mu= random('Normal',0,sigma_mu,size(r));

for i= 2:num_trial
    if i==50
        w(i)= w(i-1)+ delta_w(i-1)+ mu(i-1) -3;
    elseif i==80
        w(i)= w(i-1)+ delta_w(i-1)+ mu(i-1)+ 6;
    else
        w(i)= w(i-1)+ delta_w(i-1)+ + mu(i-1);
    end
    r(i)= w(i)*u+delta_r(i-1);
    
end

figure
subplot(3,2,1)
scatter(1:num_trial,w,'filled','MarkerEdgeColor','k','MarkerFaceColor','k')
hold on
scatter(1:num_trial,r,70.*ones(size(r)),'x','MarkerEdgeColor','k')
hold on
viscircles([50 w(50)],1.5,'color','k','LineWidth',0.5);
viscircles([80 w(80)],1.5,'color','k','LineWidth',0.5);
lgd= legend('r(t)','w(t)','interpreter','latex','location','best'); lgd.FontSize= 10;
xticks([0 50 100])
yticks([-4 -2 0 2 4])
a= get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',16,'fontweight','bold')
a= get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',16,'fontweight','bold')



w_hat= zeros(1,num_trial); c= zeros(1,num_trial);  MSE= zeros(1,num_trial);
beta= zeros(1,num_trial); sigma= zeros(1,num_trial); delta_w=[];
sigma_phi= 0.001;
u=1; tau= 0.9; sigma(1)= 100; threshold= 3.3;
phi= random('Normal',0,sigma_phi,size(c));


for i= 2:num_trial
    delta_sigma= sigma(i-1)*u/(u*sigma(i-1)*u+tau^2)*u*sigma(i-1);
    sigma(i)= sigma(i-1)-delta_sigma;
    
    delta_w= (r(i-1)-u*w_hat(i-1)).*(sigma(i-1)*u./(u*sigma(i-1)*u+tau^2));
    w_hat(i)= w_hat(i-1)+delta_w+c(i-1)*phi(i-1);
    
    beta(i)= (r(i)-u*w_hat(i))^2/(u*sigma(i-1)*u+tau^2);
    if beta(i)>=threshold
        c(i)= 1;  sigma(i)= 100;
    end
    MSE(i)= (w_hat(i)-w(i)).^2;
    
end


%figure
subplot(3,2,3)
scatter(1:num_trial,w_hat,'filled','MarkerEdgeColor','k','MarkerFaceColor','k')
hold on
scatter(1:num_trial,r,70.*ones(size(r)),'x','MarkerEdgeColor','k')
lgd= legend('r(t)','$\hat{w}(t)$','interpreter','latex','location','best'); lgd.FontSize= 10;
xticks([0 50 100])
yticks([-4 -2 0 2 4])
a= get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',16,'fontweight','bold')
a= get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',16,'fontweight','bold')

%figure
subplot(3,2,5)
plot(1:num_trial,sigma,'--k')
hold on
plot(1:num_trial,beta,'k')
hold on
yline(threshold,'--')
ylim([0 10])
lgd= legend('ACH','NE','$\gamma$','interpreter','latex','location','northwest');
lgd.FontSize= 10;
xticks([0 50 100])
yticks([0 5 10])
a= get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',16,'fontweight','bold')
a= get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',16,'fontweight','bold')

threshold= 0:0.3:15; iter=1000; MSE= zeros(length(threshold),iter,num_trial); 
num_trial= 100; sigma_w= 0.1; sigma_r= 0.6; sigma_mu= 0.01;  u= 1;
sigma_phi= 0.01;
tau= 0.9; sigma(1)= 100;

for k=1:length(threshold)
    for j=1:iter
        r= zeros(1,num_trial); w= zeros(1,num_trial);
        delta_r= random('Normal',0,sigma_r,size(r)); delta_w= random('Normal',0,sigma_w,size(w));
        
        for i= 2:num_trial
            if i==50
                w(i)= w(i-1)+ delta_w(i-1)+ mu(i-1) -3;
            elseif i==80
                w(i)= w(i-1)+ delta_w(i-1)+ mu(i-1)+ 6;
            else
                w(i)= w(i-1)+ delta_w(i-1)+ mu(i-1);
            end
            r(i)= w(i)*u+delta_r(i-1);
            
        end
        
        
        w_hat= zeros(1,num_trial); c= zeros(1,num_trial);  
        beta= zeros(1,num_trial); sigma= zeros(1,num_trial); delta_w=[];
        phi= random('Normal',0,sigma_phi,size(c));
        
        
        for i= 2:num_trial
            delta_sigma= sigma(i-1)*u/(u*sigma(i-1)*u+tau^2)*u*sigma(i-1);
            sigma(i)= sigma(i-1)-delta_sigma;
            
            delta_w= (r(i-1)-u*w_hat(i-1)).*(sigma(i-1)*u./(u*sigma(i-1)*u+tau^2));
            w_hat(i)= w_hat(i-1)+delta_w+c(i-1)*phi(i-1);
            
            beta(i)= (r(i)-u*w_hat(i))^2/(u*sigma(i-1)*u+tau^2);
            if beta(i)>= threshold(k)
                c(i)= 1;  sigma(i)= 100;
            end
            MSE(k,j,i)= (w_hat(i)-w(i)).^2;
        end
    end
end


mse= sum(MSE,3);

%figure
subplot(3,2,[2,4,6])
tmp= mean(mse,2);
e= errorbar(threshold,tmp,var(mse,[],2)/iter);
e.Color = 'k';
hold on
yline(tmp(tmp==min(tmp,[],'all')))
hold on
ylabel('MSE','interpreter','latex');
xlabel('$\gamma$','interpreter','latex')
yline(tmp(1))
xticks([0 threshold(find(tmp==min(tmp,[],'all'))) 15])
yticks([0,20,40,60,80,100])
a= get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',10,'fontweight','bold')
a= get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',10,'fontweight','bold')
