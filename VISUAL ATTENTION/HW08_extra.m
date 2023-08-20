close all
clear
clc

datafolder = 'D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW08\data\Eye tracking database\Eye tracking database\DATA/hp';
stimfolder = 'D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW08\data\Eye tracking database\Eye tracking database\ALLSTIMULI';
files=dir(fullfile(stimfolder,'*.jpeg'));
[filenames{1:size(files,1)}] = deal(files.name);
%%

%%%% This part I add feature to model at each steps. Be aware, it takes long
%%%% time.
% Subband feature

for i= 1:320
    clc
    saliencyMap_Subband{i} = saliency(filenames{1,i});
end
%%

% Torralba & Subband feature

for i= 1:320
    clc
    saliencyMap_Torralba_Subband{i} = saliency(filenames{1,i});
end
%%
% Torralba & Subband & Colour feature

for i= 1:320
    clc
    saliencyMap_Torralba_Subband_Colour{i} = saliency(filenames{1,i});
end

%%
% Torralba & Subband & Colour & Horizon & Object feature

for i= 1:320
    clc
    i
    saliencyMap_Torralba_Subband_Colour_Object{i} = saliency(filenames{1,i});
end
%%
% Torralba & Subband & Colour & Horizon & Object & Itti feature

for i= 1:320
    clc
    i
    saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti{i} = saliency(filenames{1,i});
end
%%
clc
load('saliencyMap_Subband.mat');
%%
% up-down
load('saliencyMap_Torralba_Subband.mat');
load('saliencyMap_Torralba_Subband_Colour.mat');
load('saliencyMap_Torralba_Subband_Colour_Object.mat');
load('saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti.mat');
%%
load('saliencyMap_Subband.mat');
%%
% intact
Intact_saliencyMap_Subband= struct2cell(load('saliency-subband'));
%%
Intact_saliencyMap_Torralba_Subband= struct2cell(load('saliencyMap_Torralba_Subband'));
Intact_saliencyMap_Torralba_Subband_Colour= struct2cell(load('saliencyMap_Torralba_Subband_Colour'));
Intact_saliencyMap_Torralba_Subband_Colour_Horizon_Object= struct2cell(load('saliencyMap_Torralba_Subband_Colour_Horizon_Object'));
Intact_saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti= struct2cell(load('saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti'));

%%
img_num= 80; %40 %70 %80 90

img= imread(fullfile(stimfolder,filenames{img_num}));
figure
subplot(532)
imshow(img); hold on;
eyeData= hp_data{1, img_num}.eyeData; 
fixs= hp_data{1, img_num}.fixs;
plot(eyeData(:,1),eyeData(:,2),'r.','MarkerSize',10); %Plot all data points (red dots)
plot(eyeData(fixs,1),eyeData(fixs,2),'y.','MarkerSize',5); %Plot all fixations (yellow dots)
appropFix = floor(Fix{img_num}.medianXY(2:end, :));  % we start at fixation 2

for j = 1:length(appropFix)
    %if j~=3
    text (appropFix(j, 1), appropFix(j, 2), ['{\color{black}\bf', num2str(j), '}'], 'FontSize', 5,...
        'BackgroundColor', [1, 1, 0]);
    %end
end
title('Original Image','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])

subplot(634)
imagesc(Intact_saliencyMap_Subband{1,1}{1,img_num})
title('saliencyMap\_Subband (intact image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(636)
imagesc(flipud(saliency_Map{1,img_num}))
title('saliencyMap\_Subband (up to down image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(637)
imagesc(Intact_saliencyMap_Torralba_Subband{1,1}{1,img_num})
title('saliencyMap\_Subband\_Torralba (intact image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(639)
imagesc(flipud(saliencyMap_Torralba_Subband{1,img_num}))
title('saliencyMap\_Subband\_Torralba (up to down image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(6,3,10)
imagesc(Intact_saliencyMap_Torralba_Subband_Colour{1,1}{1,img_num})
title('saliencyMap\_Subband\_Torralba\_Colour (intact image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(6,3,12)
imagesc(flipud(saliencyMap_Torralba_Subband_Colour{1,img_num}))
title('saliencyMap\_Subband\_Torralba\_Colour (up to down image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(6,3,13)
imagesc(Intact_saliencyMap_Torralba_Subband_Colour_Horizon_Object{1,1}{1,img_num})
title('saliencyMap\_Subband\_Torralba\_Colour\_Horizon\_Object (intact image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(6,3,15)
imagesc(flipud(saliencyMap_Torralba_Subband_Colour_Object{1,img_num}))
title('saliencyMap\_Subband\_Torralba\_Colour\_Horizon\_Object (up to down image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(6,3,16)
imagesc(Intact_saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti{1,1}{1,img_num})
title('saliencyMap\_Subband\_Torralba\_Colour\_Horizon\_Object\_Itti (intact image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
subplot(6,3,18)
imagesc(flipud(saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti{1,img_num}))
title('saliencyMap\_Subband\_Torralba\_Colour\_Horizon\_Object\_Itti (up to down image)','interpreter','latex')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
colormap jet

%%
% Subband
saliency_Map= saliencyMap_Subband;
a_Subband= ROC_area_func(flipud(saliency_Map),hp_data);

% Torralba_Subband
saliency_Map= saliencyMap_Torralba_Subband;
a_Torralba_Subband= ROC_area_func(flipud(saliency_Map),hp_data);

% Torralba_Subband_Colour
saliency_Map= saliencyMap_Torralba_Subband_Colour;
a_Torralba_Subband_Colour= ROC_area_func(flipud(saliency_Map),hp_data);

% Torralba_Subband_Colour_Horizon_Object
saliency_Map= saliencyMap_Torralba_Subband_Colour_Object;
a_Torralba_Subband_Colour_Object= ROC_area_func(flipud(saliency_Map),hp_data);

% Torralba_Subband_Colour_Object_Itti
saliency_Map= saliencyMap_Torralba_Subband_Colour_Horizon_Object_Itti;
a_Torralba_Subband_Colour_Object_Itti= ROC_area_func(flipud(saliency_Map),hp_data);
%%
clc
mean_a_Subband= mean(a_Subband,1);
mean_a_Torralba_Subband= mean(a_Torralba_Subband,1);
mean_a_Torralba_Subband_Colour= mean(a_Torralba_Subband_Colour,1);
mean_a_Torralba_Subband_Colour_Object= mean(a_Torralba_Subband_Colour_Object,1);
mean_a_Torralba_Subband_Colour_Object_Itti= mean(a_Torralba_Subband_Colour_Object_Itti,1);


M1= [mean_a_Subband(1),mean_a_Torralba_Subband(1),...
    mean_a_Torralba_Subband_Colour(1),...
    mean_a_Torralba_Subband_Colour_Object(1),mean_a_Torralba_Subband_Colour_Object_Itti(1)];

M2= [mean_a_Subband(2),mean_a_Torralba_Subband(2),...
    mean_a_Torralba_Subband_Colour(2),...
    mean_a_Torralba_Subband_Colour_Object(2),mean_a_Torralba_Subband_Colour_Object_Itti(2)];
%%
figure
plot(M1,'linewidth',1.5); hold on; plot(M2,'linewidth',1.5);
hold on
scatter(1:5,M1,'fill'); hold on; scatter(1:5,M2,'fill');
legend('bottom-up','top-down','interpreter','latex')
xticks([1 2 3 4 5])
xticklabels({'Subband Features','Subband\_Torralba Features',...
    'Torralba\_Subband\_Colour Features',...
    'Torralba\_Subband\_Colour\_Object Features','Torralba\_Subband\_Colour\_Object\_Itti Features'})
ylabel('ROC Value','interpreter','latex')

ax = gca;
ax. TickLabelInterpreter= 'latex'; 
xlim([0 6])

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
