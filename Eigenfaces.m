%% Eigenfaces
%%
clc
clear
%close all

%Q3

Mp = 15;
T = 30;

Pt = 'D:\Zahra\matlabNeuron\Test';
[Test,Testcell] = loadfile(Pt);
%imshow(Testcell{1,1})
%imshow(Testcell{25,1})
%imshow(Testcell{29,1})
%%
for t = 1:30
[omega{t},Q{t}] = DetectFace(Testcell,t,Mp,T);
end
%%
Mp = 5;
Pt = 'D:\Zahra\matlabNeuron\Classify';
[Test,Testcell] = loadfile(Pt);
for t = 1:30
[omegaK{t},QK{t}] = DetectFace(Testcell,t,Mp,T);
end

T = 10;
Pt = 'D:\Zahra\matlabNeuron\Detect';
[Test,Testcell] = loadfile(Pt);
for t = 1:10
[omegaT{t},QT{t}] = DetectFace(Testcell,t,Mp,T);
end

%%
Mp = 1:10;
teta = 3e+3;

for i = 1:length(Mp)
accurancy(i) = acuurency_find(Mp(i),teta);
end
plot(Mp,accurancy);
xlabel('M')
ylabel('accurancy')
%%
% Non Face
clc
close all
clear

I{1,1} = imread('1.jpg');
I{2,1} = imread('2.jpg');
I{3,1} = imread('3.jpg');


Pt = 'D:\Zahra\matlabNeuron\Test';
[Test,Testcell] = loadfile(Pt);

T = 3;
for t = 1:T
    w{t} = double(I{t,1});
end
S=sum(cat(3,w{:}),3);
m = S./T;
%figure
%imshow(uint8(m));
%title('Mean Face Image');

for t = 1:T
   meanpic{t} = w{t} - m;
end


Mp = 30;
T = 30;
s = 0;
for t = 1:30
[omegaT{t},Q{t},V4] = DetectFace(Testcell,t,Mp,T);
end

for in = 1:3
meanpic1 = reshape(meanpic{1,in},[],1);
for i=1:Mp
    omega(i) = V4{1,i}'*meanpic1;
    s = s + omega(i)*V4{1,i};
end

m1 = reshape(m,[],1);
recupic = s + m1;
Q = reshape(recupic,[],92);

figure
imshow(uint8(Q))
title('Reconstructed image')
end

%%
T = 30;
Pt = 'D:\Zahra\matlabNeuron\Classify';
[Test,Testcell] = loadfile(Pt);
for t = 1:30
[omegaK{t},QK{t}] = DetectFace(Testcell,t,Mp,T);
end

N(1,1) = 0;
teta = 3000;

for t = 1:3
   [i,eps{t}] = findClassify(omegaT{1,t},omegaK,teta);
   if eps{1,t} ~=0
       N(1,t) = i;
   end
end

accurancy = (nnz(N))/40*100
%%






function [c,C] = loadfile(P)
D = dir(fullfile(P,'*.pgm'));
C = cell(size(D));
for k = 1:numel(D)
    C{k} = imread(fullfile(P,D(k).name));
end
c = double(cell2mat(C));
end
function [I,M] = findClassify(omegaT,omegaK,teta)
for t = 1:30
   eps(t) =  norm(omegaT-omegaK{1,t});
end    
[M,I] = min(eps);
%if M > 3e+3
if M > teta
    M = 0;
end
end
function [omega, Q,eignface] = DetectFace(Testcell,in,Mp,T)
for t = 1:T
    w{t} = double(Testcell{t,1});
end
S=sum(cat(3,w{:}),3);
m = S./T;
%figure
%imshow(uint8(m));
%title('Mean Face Image');

for t = 1:T
   meanpic{t} = w{t} - m;
   repw{t} = reshape(meanpic{1,t},1,[]);
end

%
%{
figure
imshow(meanpic{1,15});
title('The face image which mean was reduced')
figure
imshow(meanpic{1,10});
title('The face image which mean was reduced')
figure
imshow(meanpic{1,20});
title('The face image which mean was reduced')
figure
imshow(meanpic{1,30});
title('The face image which mean was reduced')
%}
%
A = repw{1,1}';
for t = 2:T
    A = cat(2,A,repw{1,t}');
end
At = transpose(A)*A;
[V,D] = eig(At);
V = A*V;

for t = 1:T
V1(:,t) = V(:,t)./norm(V(:,t));
end

x = diag(D);
[B,I] = sort(x);

for t = 1:T
   %V2(:,t) = V(:,I(31-t,1)); %instead of V1
   V2(:,t) = V1(:,I((T+1)-t,1));
end

for t = 1:T
V3{t} = reshape(V2(:,t),[],92);
end
%{
figure
imagesc(V3{1,1});
title('The key eignface image');
%}

for t = 1:Mp
V4{t} = V2(:,t);
end

eignface = V4;
s = 0;

meanpic1 = reshape(meanpic{1,in},[],1);
for i=1:Mp
    omega(i) = V4{1,i}'*meanpic1;
    s = s + omega(i)*V4{1,i};
end

m1 = reshape(m,[],1);
recupic = s + m1;
Q = reshape(recupic,[],92);

%{
figure
imshow(uint8(Q))
title('Reconstructed image')
%}
end
function accurancy = acuurency_find(Mp,teta)
T = 30;
Pt = 'D:\Zahra\matlabNeuron\Classify';
[Test,Testcell] = loadfile(Pt);
for t = 1:30
[omegaK{t},QK{t}] = DetectFace(Testcell,t,Mp,T);
end

T = 10;
Pt = 'D:\Zahra\matlabNeuron\Detect';
[Test,Testcell] = loadfile(Pt);
for t = 1:10
[omegaT{t},QT{t}] = DetectFace(Testcell,t,Mp,T);
end
%
close all
for t = 1:10
   [i,eps{t}] = findClassify(omegaT{1,t},omegaK,teta);
   if eps{1,t} ~=0
       I(1,t) = i;
   end
   
end  
accurancy = (nnz(I))/40*100;
end