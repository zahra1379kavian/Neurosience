close all
clear
clc

% load data and plot
n= 5;
load('IMAGES.mat')
figure
imshow(images(:,:,n))

%%
% pre-process images
images_size = size(IMAGES,1);
num_images = size(IMAGES,3);

images = pre_process_images(images_size,num_images,IMAGES);

save IMAGES images

figure
imshow(images(:,:,n))

%% Natural Images
clc
close all
clear

load('IMAGES.mat')
%load('IMAGES_main.mat');
A = rand(256)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

% plot the bases function and related noise variance
figure(1)
title('Sparse Basis Function','interpreter','latex')
colormap(gray)
sparsenet_main(images,A);
%sparsenet_main(IMAGES,A);

%% Yale Datasets
% load different images dataset and find the bases function and noise
% distribution
close all
clc
clear

stimfolder= 'D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW09\sparsenet\images_dataset\Yale_face\data';
files= dir(fullfile(stimfolder));

label= randi([3 length(files)],1,10);
images= zeros(200,200,10);
images_colormap= zeros(256,3,10);

for i= 1:length(label)
    importfile(files(label(i)).name);
    images(:,:,i)= imresize(cdata,[200 200]);
    images_colormap(:,:,i)= colormap;
end

%save prefered_images images
clear colormap

% for i = 1:10
%     figure
%     imagesc(images(:,:,i));
%     colormap(images_colormap(:,:,i))
%     
% end


clc
% clean images in datasets
images_size = size(images,1);
num_images = size(images,3);
images_cleaned = pre_process_images(images_size,num_images,images);


% create basis function matrix
A = rand(256)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

% plot the bases function and related noise variance
figure(1)
title('Sparse Basis Function','interpreter','latex')
colormap(gray)
sparsenet_main(images_cleaned,A);
%% Caltech Datasets
close all
clc
clear

stimfolder= 'D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW09\sparsenet\images_dataset\Caltech101\my_images';
files=dir(fullfile(stimfolder,'*.jpg'));
[filenames{1:size(files,1)}] = deal(files.name);


%label= randi([3 length(files)],1,10);
label= [100 99 98 97 96 110 109 108 107 106];
images= zeros(200,200,10);
images_colormap= zeros(256,3,10);

for i= 1:length(label)
    %files(label(i)).name;
    cdata= rgb2gray(imread(fullfile(stimfolder,filenames{label(i)})));
    images(:,:,i)= imresize(cdata,[200 200]);
end


%save prefered_images images

% close all
% for i = 1:10
%     figure
%     imagesc(images(:,:,i));
%     colormap gray
% end


clc
% clean images in datasets
images_size = size(images,1);
num_images = size(images,3);
images_cleaned = pre_process_images(images_size,num_images,images);


% create basis function matrix
A = rand(256)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

% plot the bases function and related noise variance
figure(1)
title('Sparse Basis Function','interpreter','latex')
colormap(gray)
sparsenet_main(images_cleaned,A);


%% MNIST datasets
close all
clc
clear

filename= 'train-images.idx3-ubyte';
IMAGES = loadMNISTImages(filename);

for i= 1:size(IMAGES,1)
    images(:,:,i)=  reshape(IMAGES(:,i),28,28);
    
end


% for i = 1:10
%     figure
%     imagesc(images(:,:,i));
%     colormap gray
% end

% clean images in datasets
images_size = size(images,1);
num_images = size(images,3);
images_cleaned = pre_process_images(images_size,num_images,images);


% create basis function matrix
A = rand(256)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

% plot the bases function and related noise variance
figure(1)
title('Sparse Basis Function','interpreter','latex')
colormap(gray)
sparsenet_main(images_cleaned,A);

%% Video Farme
close all
clc
clear

movieInfo = VideoReader('BIRD.avi');
numOfFrame = movieInfo.NumFrames;

Frames = read(movieInfo,[1,10]);

figure('Position', [10 50 1500 700]);
colormap gray
count= 1; count_frame=1; %myMovie= {480,1};

for i= 1:10:numOfFrame-9
    Frames= read(movieInfo,[i,i+9]);
    for j= 1:10
        Frames_gray = rgb2gray(Frames(:,:,:,j));
        size_img= min(size(Frames_gray));
        images(:,:,j) = imresize(Frames_gray,[size_img, size_img]);
    end
    images_size = size(images,1);
    num_images = size(images,3);
    images_cleaned = pre_process_images(images_size,num_images,images);
    
    A = rand(64)-0.5;
    A = A*diag(1./sqrt(sum(A.*A)));
    
    % plot the bases function and related noise variance
    %     figure(1)
    %     title('Sparse Basis Function','interpreter','latex')
    %     colormap gray
    
    if i==1
        [S,myMovie,count_frame]= sparsenet(images_cleaned,A,count,count_frame);
    else
        [S,myMovie,count_frame]= sparsenet(images_cleaned,A,count,count_frame,myMovie);
    end
    count= count+1;
end


writerObj = VideoWriter("demo2");
writerObj.FrameRate = 50;
open(writerObj);
numberOfFrames = length(myMovie);
for frameNumber = 1 : numberOfFrames
    writeVideo(writerObj, myMovie(frameNumber));
end
close(writerObj);

clc
for i = 1:11
    figure
    imshow(myMovie(i).cdata)
end
%% Optional
close all
clc
clear

load('IMAGES.mat')
for i= 1:size(images,3)
    saliencyMap{i} = saliency(images(:,:,i));
end
%%
clc
for i= 1:10
figure
subplot(121); imshow(images(:,:,i)); title('Original Image');
subplot(122); imshow(saliencyMap{1, i}); title('SaliencyMap');colormap('gray');
end

%%

for i= 1:10
   image_salienc(:,:,i)= saliencyMap{i};  
end
%%
% create basis function matrix
A = rand(64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

% plot the bases function and related noise variance
figure(1)
title('Sparse Basis Function','interpreter','latex')
colormap(gray)
sparsenet_main(image_salienc,A);








