close all
clear
clc

stimfolder = 'D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW08\data\Eye tracking database\Eye tracking database\ALLSTIMULI';
files=dir(fullfile(stimfolder,'*.jpeg'));
[filenames{1:size(files,1)}] = deal(files.name);

%%
%%%% This part I run saliency with one feature at a times. Be aware, it takes long
%%%% time.
% Whole features
% for i= 1:size(filenames,2)
%     saliencyMap{i} = saliency(filenames{1,i});
% end
% 
% %%
% % Subband feature
% for i= 1:320
%     clc
%     saliencyMap_Subband{i} = saliency(filenames{1,i});
% end
% 
% %%
% % Itti feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Itti{i} = saliency(filenames{1,i});
% end
% %%
% % Colour feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Colour{i} = saliency(filenames{1,i});
% end
% 
% %%
% % Torralba feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Torralba{i} = saliency(filenames{1,i});
% end
% 
% %%
% % Horizon feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Horizon{i} = saliency(filenames{1,i});
% end
% 
% %%
% % Object feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Object{i} = saliency(filenames{1,i});
% end
% 
% %%
% % DistToCenter feature
% 
% for i= 1:320
%     clc
%     saliencyMap_DistToCenter{i} = saliency(filenames{1,i});
% end

%%
clc
% I saved saliency matrix of the last part. Now, I reload them.
load('hp_data.mat');
load('saliencyMap_Colour.mat');
load('saliencyMap_DistToCenter.mat');
load('saliencyMap_Horizon.mat');
load('saliencyMap_Itti.mat');
load('saliencyMap_Object.mat');
load('saliencyMap_Torralba.mat');
load('saliency-subband.mat');
load('Fix.mat');
load('all-saliency-map.mat');

%%
clc
close all
% In this part I plot eye track map with related saliency map. 
img_num= [1,100,200,300,250];

for i= img_num
img= imread(fullfile(stimfolder,filenames{i}));

eyeData= hp_data{1, i}.eyeData; 
fixs= hp_data{1, i}.fixs;


figure;
% plot the orginal image
imshow(img); hold on;
% plot eye tracked image
plot(eyeData(:,1),eyeData(:,2),'r.','MarkerSize',14); %Plot all data points (red dots)
plot(eyeData(fixs,1),eyeData(fixs,2),'y.','MarkerSize',14); %Plot all fixations (yellow dots)
appropFix = floor(Fix{1}.medianXY(2:end, :));  % we start at fixation 2

for j = 1:length(appropFix)
    text (appropFix(j, 1), appropFix(j, 2), ['{\color{black}\bf', num2str(j), '}'], 'FontSize', 16, 'BackgroundColor', [1, 1, 0]);
end

% plot saliency map
figure;
subplot(331); 
imshow(img); title('Original Image','interpreter','latex');
subplot(332); 
imshow(saliencyMap{1,i}); title('SaliencyMap','interpreter','latex');colormap('gray');
subplot(333); 
imshow(saliencyMap_DistToCenter{1,i}); title('SaliencyMap-DistToCenter','interpreter','latex');colormap('gray');
subplot(334); 
imshow(saliencyMap_Horizon{1,i}); title('SaliencyMap-Horizon','interpreter','latex');colormap('gray');
subplot(335); 
imshow(saliencyMap_Itti{1,i}); title('SaliencyMap-Itti','interpreter','latex');colormap('gray');
subplot(336); 
imshow(saliencyMap_Object{1,i}); title('SaliencyMap-Object','interpreter','latex');colormap('gray');
subplot(337); 
imshow(saliencyMap_Subband{1,i}); title('SaliencyMap-Subband','interpreter','latex');colormap('gray');
subplot(338); 
imshow(saliencyMap_Torralba{1,i}); title('SaliencyMap-Torralba','interpreter','latex');colormap('gray');
subplot(339); 
imshow(saliencyMap_Colour{1,i}); title('SaliencyMap-Colour','interpreter','latex');colormap('gray');

end

%%
clc
% For each asliency map, I calculate the area under curve as a model
% performance
t1= round(round(size(hp_data{1, 1}.eyeData,1)/2));

% saliencyMap
saliency_Map= saliencyMap;
a= ROC_area_func(saliency_Map,hp_data);

% DistToCenter
saliency_Map= saliencyMap_DistToCenter;
a_DistToCenter= ROC_area_func(saliency_Map,hp_data);

% Horizon
saliency_Map= saliencyMap_Horizon;
a_Horizon= ROC_area_func(saliency_Map,hp_data);

% Itti
saliency_Map= saliencyMap_Itti;
a_Itti= ROC_area_func(saliency_Map,hp_data);

% Object
saliency_Map= saliencyMap_Object;
a_Object= ROC_area_func(saliency_Map,hp_data);

% Subband
saliency_Map= saliencyMap_Subband;
a_Subband= ROC_area_func(saliency_Map,hp_data);

% Torralba
saliency_Map= saliencyMap_Torralba;
a_Torralba= ROC_area_func(saliency_Map,hp_data);

% Colour
saliency_Map= saliencyMap_Colour;
a_Colour= ROC_area_func(saliency_Map,hp_data);
%%
clc
% compare top-down and bottom-up of each features group
figure
subplot(3,3,1)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a(:,2),'Normalization','pdf')
title('Whole Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,2)
histogram(a_DistToCenter(:,1),'Normalization','pdf'); hold on; histogram(a_DistToCenter(:,2),'Normalization','pdf')
title('DistToCenter Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,3)
histogram(a_Horizon(:,1),'Normalization','pdf'); hold on; histogram(a_Horizon(:,2),'Normalization','pdf')
title('Horizon Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,4)
histogram(a_Itti(:,1),'Normalization','pdf'); hold on; histogram(a_Itti(:,2),'Normalization','pdf')
title('Itti Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,5)
histogram(a_Object(:,1),'Normalization','pdf'); hold on; histogram(a_Object(:,2),'Normalization','pdf')
title('Object Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,6)
histogram(a_Subband(:,1),'Normalization','pdf'); hold on; histogram(a_Subband(:,2),'Normalization','pdf')
title('Subband Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,7)
histogram(a_Torralba(:,1),'Normalization','pdf'); hold on; histogram(a_Torralba(:,2),'Normalization','pdf')
title('Torralba Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')
subplot(3,3,8)
histogram(a_Colour(:,1),'Normalization','pdf'); hold on; histogram(a_Colour(:,2),'Normalization','pdf')
title('Colour Features','interpreter','latex')
legend('bottom-up','top-down','interpreter','latex','location','best')

sgtitle('Compare Top-down \& Bottom-up of Each Features Group','interpreter','latex')
%%
% Compare Bottom-up of whole features and each other features
figure
subplot(3,3,1)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_DistToCenter(:,1),'Normalization','pdf')
title('DistToCenter Features','interpreter','latex')
legend('Whole','DistToCenter','interpreter','latex','location','best')
subplot(3,3,2)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_Horizon(:,1),'Normalization','pdf')
title('Horizon Features','interpreter','latex')
legend('Whole','Horizon','interpreter','latex','location','best')
subplot(3,3,3)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_Itti(:,1),'Normalization','pdf')
title('Itti Features','interpreter','latex')
legend('Whole','Itti','interpreter','latex','location','best')
subplot(3,3,4)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_Object(:,1),'Normalization','pdf')
title('Object Features','interpreter','latex')
legend('Whole','Object','interpreter','latex','location','best')
subplot(3,3,5)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_Subband(:,1),'Normalization','pdf')
title('Subband Features','interpreter','latex')
legend('Whole','Subband','interpreter','latex','location','Northwest')
subplot(3,3,6)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_Torralba(:,1),'Normalization','pdf')
title('Torralba Features','interpreter','latex')
legend('Whole','Torralba','interpreter','latex','location','best')
subplot(3,3,8)
histogram(a(:,1),'Normalization','pdf'); hold on; histogram(a_Colour(:,1),'Normalization','pdf')
title('Colour Features','interpreter','latex')
legend('Whole','Colour','interpreter','latex','location','best')

sgtitle('Compare Bottom-up of Each Features Group with Whole Features','interpreter','latex')
%%
% Compare Top-down of whole features and each other features
figure
subplot(3,3,1)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_DistToCenter(:,2),'Normalization','pdf')
title('DistToCenter Features','interpreter','latex')
legend('Whole','DistToCenter','interpreter','latex','location','Northeast')
subplot(3,3,2)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_Horizon(:,2),'Normalization','pdf')
title('Horizon Features','interpreter','latex')
legend('Whole','Horizon','interpreter','latex','location','best')
subplot(3,3,3)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_Itti(:,2),'Normalization','pdf')
title('Itti Features','interpreter','latex')
legend('Whole','Itti','interpreter','latex','location','best')
subplot(3,3,4)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_Object(:,2),'Normalization','pdf')
title('Object Features','interpreter','latex')
legend('Whole','Object','interpreter','latex','location','best')
subplot(3,3,5)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_Subband(:,2),'Normalization','pdf')
title('Subband Features','interpreter','latex')
legend('Whole','Subband','interpreter','latex','location','Northwest')
subplot(3,3,6)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_Torralba(:,2),'Normalization','pdf')
title('Torralba Features','interpreter','latex')
legend('Whole','Torralba','interpreter','latex','location','Northwest')
subplot(3,3,8)
histogram(a(:,2),'Normalization','pdf'); hold on; histogram(a_Colour(:,2),'Normalization','pdf')
title('Colour Features','interpreter','latex')
legend('Whole','Colour','interpreter','latex','location','best')

sgtitle('Compare Top-down of Each Features Group with Whole Features','interpreter','latex')

%%
clc
% Average of ROC over whole images
mean_a= mean(a,1);
mean_a_DistToCenter= mean(a_DistToCenter,1);
mean_a_Horizon= mean(a_Horizon,1);
mean_a_Itti= mean(a_Itti,1);
mean_a_Object= mean(a_Object,1);
mean_a_Subband= mean(a_Subband,1);
mean_a_Torralba= mean(a_Torralba,1);
mean_a_Colour= mean(a_Colour,1);


M1= [mean_a(1),mean_a_Horizon(1),mean_a_Subband(1),mean_a_Torralba(1),...
    mean_a_Itti(1),mean_a_Colour(1),mean_a_Object(1),mean_a_DistToCenter(1)];

M2= [mean_a(2),mean_a_Horizon(2),mean_a_Subband(2),mean_a_Torralba(2),...
    mean_a_Itti(2),mean_a_Colour(2),mean_a_Object(2),mean_a_DistToCenter(2)];

% Compare the model performance for each salinecy in two time group
figure
plot(M1,'linewidth',1.5); hold on; plot(M2,'linewidth',1.5);
hold on
scatter(1:8,M1,'fill'); hold on; scatter(1:8,M2,'fill');
legend('bottom-up','top-down','interpreter','latex')
xticks([1 2 3 4 5 6 7 8])
xticklabels({'Whole Features','Horizon Features','Subband Features','Torralba Features','Itti Features','Colour Features','Object Features',...
   'DistToCenter Features'})
ylabel('ROC value','interpreter','latex')
ax = gca;
ax. TickLabelInterpreter= 'latex'; 
xlim([0 9])
xtickangle(45)
title('Mean of ROC for Each Features','interpreter','latex','FontSize',15)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%% This part I add feature to model at each steps. Be aware, it takes long
%%%% time.

% Torralba & Subband feature

% for i= 1:320
%     clc
%     saliencyMap_Torralba_Subband{i} = saliency(filenames{1,i});
% end
% 
% % Torralba & Subband & Colour feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Torralba_Subband_Colour{i} = saliency(filenames{1,i});
% end
% 
% 
% % Torralba & Subband & Colour & Horizon feature
% 
% for i= 1:320
%     clc
%     saliencyMap_Torralba_Subband_Colour_Horizon{i} = saliency(filenames{1,i});
% end
% 
% % Torralba & Subband & Colour & Horizon & Object feature
% 
% for i= 1:320
%     clc
%     i
%     saliencyMap_Torralba_Subband_Colour_Horizon_Object{i} = saliency(filenames{1,i});
% end
% 
% % Torralba & Subband & Colour & Horizon & Object & Itti feature
% 
% for i= 1:320
%     clc
%     i
%     saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti{i} = saliency(filenames{1,i});
% end

%%
% I reload the last part saliency map
clc
load('hp_data');
load('all-saliency-map');
load('saliency-subband');
load('saliencyMap_Torralba_Subband');
load('saliencyMap_Torralba_Subband_Colour');
load('saliencyMap_Torralba_Subband_Colour_Horizon');
load('saliencyMap_Torralba_Subband_Colour_Horizon_Object');
load('saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti');
%%
% Again, calculate the ROC of each map
% saliencyMap
saliency_Map= saliencyMap;
a= ROC_area_func(saliency_Map,hp_data);

% Subband
saliency_Map= saliencyMap_Subband;
a_Subband= ROC_area_func(saliency_Map,hp_data);

% Torralba_Subband
saliency_Map= saliencyMap_Torralba_Subband;
a_Torralba_Subband= ROC_area_func(saliency_Map,hp_data);

% Torralba_Subband_Colour
saliency_Map= saliencyMap_Torralba_Subband_Colour;
a_Torralba_Subband_Colour= ROC_area_func(saliency_Map,hp_data);

% Torralba_Subband_Colour_Horizon
saliency_Map= saliencyMap_Torralba_Subband_Colour_Horizon;
a_Torralba_Subband_Colour_Horizon= ROC_area_func(saliency_Map,hp_data);

% Torralba_Subband_Colour_Horizon_Object
saliency_Map= saliencyMap_Torralba_Subband_Colour_Horizon_Object;
a_Torralba_Subband_Colour_Horizon_Object= ROC_area_func(saliency_Map,hp_data);

% Torralba_Subband_Colour_Horizon_Object_Itti
saliency_Map= saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti;
a_Torralba_Subband_Colour_Horizon_Object_Itti= ROC_area_func(saliency_Map,hp_data);

%%
clc
% Average of ROC over all images 
mean_a_Subband= mean(a_Subband,1);
mean_a_Torralba_Subband= mean(a_Torralba_Subband,1);
mean_a_Torralba_Subband_Colour= mean(a_Torralba_Subband_Colour,1);
mean_a_Torralba_Subband_Colour_Horizon= mean(a_Torralba_Subband_Colour_Horizon,1);
mean_a_Torralba_Subband_Colour_Horizon_Object= mean(a_Torralba_Subband_Colour_Horizon_Object,1);
mean_a_Torralba_Subband_Colour_Horizon_Object_Itti= mean(a_Torralba_Subband_Colour_Horizon_Object_Itti,1);


M1= [mean_a_Subband(1),mean_a_Torralba_Subband(1),...
    mean_a_Torralba_Subband_Colour(1),mean_a_Torralba_Subband_Colour_Horizon(1),...
    mean_a_Torralba_Subband_Colour_Horizon_Object(1),mean_a_Torralba_Subband_Colour_Horizon_Object_Itti(1)];

M2= [mean_a_Subband(2),mean_a_Torralba_Subband(2),...
    mean_a_Torralba_Subband_Colour(2),mean_a_Torralba_Subband_Colour_Horizon(2),...
    mean_a_Torralba_Subband_Colour_Horizon_Object(2),mean_a_Torralba_Subband_Colour_Horizon_Object_Itti(2)];

% Compare the model performance for each group of features
figure
plot(M1,'linewidth',1.5); hold on; plot(M2,'linewidth',1.5);
hold on
scatter(1:6,M1,'fill'); hold on; scatter(1:6,M2,'fill');
legend('bottom-up','top-down','interpreter','latex')
xticks([1 2 3 4 5 6])
xticklabels({'Subband Features','Subband\_Torralba Features',...
    'Torralba\_Subband\_Colour Features','Torralba\_Subband\_Colour\_Horizon Features',...
    'Torralba\_Subband\_Colour\_Horizon\_Object Features','Torralba\_Subband\_Colour\_Horizon\_Object\_Itti Features'})
ylabel('ROC Value','interpreter','latex')

ax = gca;
ax. TickLabelInterpreter= 'latex'; 
xlim([0 7])

xtickangle(45)
title('Mean of ROC for Each Models','interpreter','latex','FontSize',15)

%% Function
function a= ROC_area_func(saliency_Map,hp_data)
t1= round(round(size(hp_data{1, 1}.eyeData,1)/2));

for i= 1:size(saliency_Map,2)
    for j= 1:2
        if j== 1
            salmap= saliency_Map{1,i};
            X= hp_data{1, i}.eyeData(1:t1,1);
            Y= hp_data{1, i}.eyeData(1:t1,2);
            origimgsize= size(salmap);
            
            a(i,j) = rocScoreSaliencyVsFixations( salmap , X' , Y' , origimgsize );
        else
            salmap= saliency_Map{1,i};
            X= hp_data{1, i}.eyeData(t1+1:end,1);
            Y= hp_data{1, i}.eyeData(t1+1:end,2);
            origimgsize= size(salmap);
            
            a(i,j) = rocScoreSaliencyVsFixations( salmap , X' , Y' , origimgsize );
            
        end

    end
end

end