close all
clc
clear

%%%part 1.1-a

x1 = [1 1 1 1]';
x2 = [1 1 1 -1]';
x3 = [1 1 -1 -1]';

x = [x1';x2';x3'];

w = zeros(4,4);
p = x;

for j = 1:4
    for i = 1:4
            for k = 1:3
                w(j,i) = w(j,i)+ p(k,i)*p(k,j);
            end
    end
end
 
w = w/3;
%%
%%%%part 1.1-b
s = [1;-1;-1;1];
s = sign(w*s);
s

%%
%%%part 2.1-a
clear
clc

p = 'D:\Zahra\نوروساینس\tamrin\5\train';
D = dir(fullfile(p,'*.png'));
C = cell(size(D));
for k = 1:numel(D)
    C{k} = imread(fullfile(p,D(k).name));
end

figure

for i = 1:6
    subplot(2,3,i)
    imagesc(double(C{i,1}));
    axis on
end


%%
%%%part 2.1-a
clc
for i = 1:size(C,1)
    P(:,i) = double(reshape(C{i,1},[],1));
end

for i = 1:size(P,1)
    for j = 1:size(P,2)
        if(P(i,j) == 255)
            P(i,j) = 1;
        else
            P(i,j) = -1;
        end
    end
end
%%
%%%part 2.1-b-(weight matrix)

%%%%%%%%%%%%%%%%%%%Runnig may take long time(not too much)
w = zeros(size(P,1),size(P,1));
for j = 1:size(P,1)
    for i = 1:size(P,1)
            for k = 1:size(P,2)
                w(j,i) = w(j,i)+ P(i,k)*P(j,k);
            end
    end
end

figure
imagesc(w)

%%
%%%part 2.1-c
figure 

for i = 1:6
   pnew = sign(w*P(:,i));
   subplot(2,3,i)
   imagesc(reshape(pnew,128,[]));
   error_pnew(i,:) = pnew - P(:,i);
end

error = mean(error_pnew,2)
%%
%%%part 2.1-d(noisy images)
P1 = P;
N1 = 3000;
N2 = 8000;

%idx = randi(size(P1,1),N1,1);
idx = randi(size(P1,1),N2,1);

for j = 1:6
for i = 1:size(idx,1)
    P1(idx(i,1),j) = -1*P1(idx(i,1),j);
end
end

for i = 1:6
    for j = 1:size(P1,1)
        if P1(j,i) == 1
            P1(j,i) = 255;
        else
            P1(j,i) = 0;
        end
    end
   Cnew{i} = reshape(P1(:,i),128,[]);
end

figure

for i = 1:6
    subplot(2,3,i)
    imagesc(double(Cnew{1,i}));
    axis on
end
%%
%%%part 2.1-e(correlation)

for i = 1:6
   for j = 1:6
       x1 = double(C{i,1});
       x2 = Cnew{1,j};
       correlation(i,j) = corr2(x1,x2);
   end
end
%%
%%%part 2.1-f

figure 

for i = 1:6
   pnew = sign(w*P1(:,i));
   subplot(2,3,i)
   imagesc(reshape(pnew,128,[]));
   error_noise(i,:) = pnew - P1(:,i);
end

error_noise = mean(error_noise,2)