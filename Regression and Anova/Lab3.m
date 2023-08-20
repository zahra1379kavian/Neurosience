close all
clc
clear 

data= readtable('SearchTime.xlsx');
%% Multiple Linear Regression
clc

y= data.SearchTime; 
% DS= categorical(data.DS);
% TD= categorical(data.TD);
% X= table(DS, TD, y);
% modelspec = 'y ~ 1 + DS + TD';
% mdl = fitlm(X,modelspec);

X=[data.DS data.TD]; 
mdl = fitlm(X,y,'VarNames',{'DS','TD','Search_Time'})


%%
% Static model properties
anova(mdl,'summary') % Statics for whole model

anova(mdl) % For each variable in the model

% plotSlice(mdl)  %%%% check it

%%
% confidence interval

% Plot y and 95% confidence interval
clc
DS3= find(data.DS(:)==3);
DS5= find(data.DS(:)==5);
DS7= find(data.DS(:)==7);
DS9= find(data.DS(:)==9);

TD1= find(data.TD(:)==1);
TD2= find(data.TD(:)==3);
TD4= find(data.TD(:)==4);
TD5= find(data.TD(:)==5);

fit_DS3= mdl.Fitted(DS3);
fit_DS5= mdl.Fitted(DS5);
fit_DS7= mdl.Fitted(DS7);
fit_DS9= mdl.Fitted(DS9);

fit_TD1= mdl.Fitted(TD1);
fit_TD3= mdl.Fitted(TD3);
fit_TD4= mdl.Fitted(TD4);
fit_TD5= mdl.Fitted(TD5);

DS= [3,5,7,9];
TD= [1,3,4,5];
x= unique(data.DS)';
y= [mean(fit_DS3),mean(fit_DS5),mean(fit_DS7),mean(fit_DS9)];

% 95 % interval
ci= coefCI(mdl);
yL= ci(1,1)+ci(2,1).*DS+ci(3,1).*mean(TD);
yH= ci(1,2)+ci(2,2).*DS+ci(3,2).*mean(TD);



fig= figure;
subplot(1,2,1)
p= fill([x x(end:-1:1)],[yH yL(end:-1:1)],'b');
p.FaceColor = [201 235 243]./255;      
p.EdgeColor = 'none'; 
hold on
plot(x,y,'LineWidth',2);
hold on
plot(x,y,'bo','MarkerFaceColor','blue','LineWidth',2)
xlabel('DS','interpreter','latex')
ylabel('Fitted value','interpreter','latex')


x= unique(data.TD)';
y= [mean(fit_TD1),mean(fit_TD3),mean(fit_TD4),mean(fit_TD5)];

yL= ci(1,1)+ci(2,1).*mean(DS)+ci(3,1).*TD;
yH= ci(1,2)+ci(2,2).*mean(DS)+ci(3,2).*TD;


subplot(1,2,2)
p= fill([x x(end:-1:1)],[yH yL(end:-1:1)],'b');
p.FaceColor = [201 235 243]./255;      
p.EdgeColor = 'none'; 
hold on
plot(x,y,'LineWidth',2);
hold on
plot(x,y,'bo','MarkerFaceColor','blue','LineWidth',2)
xlabel('TD','interpreter','latex')

han=axes(fig,'visible','off');
han.Title.Visible='on';
title('95\% confidence interval','interpreter','latex')
%%

% Plot search time vs each regressor
figure
%gscatter(data.TD,data.SearchTime,data.DS,'yrgb','o+xs')
scatter(data.TD,data.SearchTime,[],'*g')
xlabel('Training Duration','Interpreter','latex'); ylabel('SearchTime','Interpreter','latex'); title('Search Time vs Training Duration','Interpreter','latex')

figure
scatter(data.DS,data.SearchTime,[],'*r')
xlabel('Disply Size','Interpreter','latex'); ylabel('SearchTime','Interpreter','latex'); title('Search Time vs Display Size','Interpreter','latex')


figure
% scatter3(X(:,1),X(:,2),y,'*r')
% hold on
x1fit = min(X(:,1)):0.1:max(X(:,1));
x2fit = min(X(:,2)):0.1:max(X(:,2));
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = mdl.Coefficients.Estimate(1)+mdl.Coefficients.Estimate(2)*X1FIT+mdl.Coefficients.Estimate(3)*X2FIT;
surface(X1FIT,X2FIT,YFIT)
xlabel('Disply Size','Interpreter','latex')
ylabel('Training Duration','Interpreter','latex')
zlabel('Search Time','Interpreter','latex')
view(3)

%% Check the quality
% Check residual normality 
figure
plotResiduals(mdl)
% figure
% plotResiduals(mdl,'probability')
figure
plotResiduals(mdl,'probability','ResidualType', 'studentized')
figure
Res = table2array(mdl.Residuals);
subplot(2,2,1)
boxplot(Res(:,1))
title('raw residual values for the model','interpreter','latex')
subplot(2,2,2)
boxplot(Res(:,2))
title('pearson residual values for the model','interpreter','latex')
subplot(2,2,3)
boxplot(Res(:,3))
title('studentized residual values for the model','interpreter','latex')
subplot(2,2,4)
boxplot(Res(:,4))
title('standardized residual values for the model','interpreter','latex')
%%
% Check the constant variance
figure
plotResiduals(mdl,'fitted','ResidualType','standardized')

figure  % Residuals against the each predictor
scatter(X(:,1),mdl.Residuals.Raw,'MarkerEdgeColor','k') 
hold on
plot(zeros(max(X(:,1))),'color','k')
xlabel('Display Size','interpreter','latex'); ylabel('Residual','interpreter','latex'); title('Residual plot against DS','interpreter','latex')

figure 
scatter(X(:,2),mdl.Residuals.Raw,'MarkerEdgeColor','k') 
hold on
plot(zeros(max(X(:,2))),'color','k')
xlabel('Training Duration','interpreter','latex'); ylabel('Residual','interpreter','latex'); title('Residual plot against TD','interpreter','latex')

figure 
scatter(data.SearchTime,mdl.Residuals.Raw,'MarkerEdgeColor','g') 
hold on
plot(data.SearchTime,data.SearchTime,'MarkerEdgeColor','black')
xlabel('Search Time','interpreter','latex'); ylabel('Residual','interpreter','latex'); title('Residual plot against search time','interpreter','latex')

%%
% Check the residual independence
figure
plotResiduals(mdl,'lagged') % Show a possible correlation among the residuals.
figure
scatter(1:size(mdl.Residuals.Raw,1),mdl.Residuals.Raw)
xlabel('Observation number','interpreter','latex')
ylabel('Residual','interpreter','latex')
title('Residual in time','interpreter','latex')
dwtest(mdl) % Null hypothesis that the residuals from a linear regression are uncorrelated.

%% Stepwise Linear Regression
clc
% First Method
X=[data.DS];
mdl= fitlm(X,y)

X= [data.TD];
mdl2= fitlm(X,mdl.Fitted)
% 

X=[data.TD];
mdl= fitlm(X,y)

X= [data.DS];
mdl2= fitlm(X,mdl2.Fitted)
%%
% Second Method
X=[data.DS];
mdl= fitlm(X,y)

X= [data.TD];
mdl2= fitlm(X,mdl.Residuals.Raw)
% 

X=[data.TD];
mdl= fitlm(X,y)

X= [data.DS];
mdl2= fitlm(X,mdl.Residuals.Raw)

%% Part 5
clc
% Histogram search time
figure
histogram(data.SearchTime) % It doesn't have normal distribution
title('Before Normalization','interpreter','latex')

pd = fitdist(data.SearchTime,'Normal');
[data_normalize,lambda] = boxcox(data.SearchTime);

 figure
histogram(data_normalize)
title('After Normalization','interpreter','latex')
% 
figure
qqplot(data_normalize,pd)
figure
qqplot(data.SearchTime)

%%
% Run model again
clc
X=[data.DS data.TD];
mdl = fitlm(X,data_normalize,'VarNames',{'DS','TD','Search_Time'})


%% Check the quality
% Check residual normality 
figure
plotResiduals(mdl)
% figure
% plotResiduals(mdl,'probability')
figure
plotResiduals(mdl,'probability','ResidualType', 'studentized')
figure
Res = table2array(mdl.Residuals);
subplot(2,2,1)
boxplot(Res(:,1))
title('raw residual values for the model','interpreter','latex')
subplot(2,2,2)
boxplot(Res(:,2))
title('pearson residual values for the model','interpreter','latex')
subplot(2,2,3)
boxplot(Res(:,3))
title('studentized residual values for the model','interpreter','latex')
subplot(2,2,4)
boxplot(Res(:,4))
title('standardized residual values for the model','interpreter','latex')
%%
% Check the constant variance
figure
plotResiduals(mdl,'fitted','ResidualType','standardized')

figure  % Residuals against the each predictor
scatter(X(:,1),mdl.Residuals.Raw,'MarkerEdgeColor','k') 
hold on
plot(zeros(max(X(:,1))),'color','k')
xlabel('Display Size','interpreter','latex'); ylabel('Residual','interpreter','latex'); title('Residual plot against DS','interpreter','latex')

figure 
scatter(X(:,2),mdl.Residuals.Raw,'MarkerEdgeColor','k') 
hold on
plot(zeros(max(X(:,2))),'color','k')
xlabel('Training Duration','interpreter','latex'); ylabel('Residual','interpreter','latex'); title('Residual plot against TD','interpreter','latex')

figure 
scatter(data.SearchTime,mdl.Residuals.Raw,'MarkerEdgeColor','g') 
hold on
plot(data.SearchTime,data.SearchTime,'MarkerEdgeColor','black')
xlabel('Search Time','interpreter','latex'); ylabel('Residual','interpreter','latex'); title('Residual plot against search time','interpreter','latex')

%%
% Check the residual independence
figure
plotResiduals(mdl,'lagged') % Show a possible correlation among the residuals.
figure
scatter(1:size(mdl.Residuals.Raw,1),mdl.Residuals.Raw)
xlabel('Observation number','interpreter','latex')
ylabel('Residual','interpreter','latex')
title('Residual in time','interpreter','latex')
dwtest(mdl) % Null hypothesis that the residuals from a linear regression are uncorrelated.
%%
clc
% Anova 
y= data.SearchTime; X=[data.DS data.TD]; 
[p,tbl2,stats] = anovan(y,{X(:,1),X(:,2)},"Model",'interaction',"Varnames",{'DS','TD'});

[results,~,~,gnames] = multcompare(stats,"Dimension",[1 2]);
tbl = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
tbl.("Group A")=gnames(tbl.("Group A"));
tbl.("Group B")=gnames(tbl.("Group B"))

%%
clc
% 'tukey-kramer' , 'scheffe', 'bonferroni'

[results,~,~,gnames] = multcompare(stats,'CType','tukey-kramer',"Dimension",[1 2]);
tbl = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
tbl.("Group A")=gnames(tbl.("Group A"));
tbl.("Group B")=gnames(tbl.("Group B"))
%%
% add subject
clc

y= data.SearchTime; X=[data.DS data.TD data.SJ]; 
[p,tbl2,stats] = anovan(y,{X(:,1),X(:,2),X(:,3)},"Model",'interaction',"Varnames",{'DS','TD','SJ'});

[results,~,~,gnames] = multcompare(stats,"Dimension",[1 2]);
tbl12 = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);

writetable(tbl12,'anova_part7_DS-TD.xlsx','sheet',1);

[results,~,~,gnames] = multcompare(stats,"Dimension",[1 3]);
tbl13 = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);

writetable(tbl13,'anova_part7_DS-Sub.xlsx','sheet',1);
 
[results,~,~,gnames] = multcompare(stats,"Dimension",[2 3]);
tbl23 = array2table(results,"VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
tbl23.("Group A")=gnames(tbl23.("Group A"));
tbl23.("Group B")=gnames(tbl23.("Group B"))
 
writetable(tbl23,'anova_part7_TD-Sub.xlsx','sheet',1);