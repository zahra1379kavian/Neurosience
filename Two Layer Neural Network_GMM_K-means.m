clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Question1
%% Neural Network
dt = 0.01;
Duration = 1000;
es = 0;%Eexc
%es = -80;%Einh
Vin = -80;%initial voltage
%Vin = -60;
a = 0.99995;
b = 0.00005;
%a = 0.99;
%b = 0.01;

T = ceil(Duration/dt);
t = (1:T) * dt;

[V,impulse_train] = Spike_Rate_Adaptation(dt,Duration,es,Vin,a,b);
figure
plot(t,V);
xlim([0 200])
title('Excitatory Synapses')
%title('Inhabitory Synapses')

%%
V1 = -80;
V2 = -60;
es = 0;
[V,impulse_train] = Spike_Rate_Adaptation(dt,Duration,es,V1,a,b);
figure
subplot(211)
plot(t,V);
hold on
[V,impulse_train] = Spike_Rate_Adaptation(dt,Duration,es,V2,a,b);
plot(t,V);
xlim([0 700])
title('Excitatory Synapses')
es = -80;
[V,impulse_train] = Spike_Rate_Adaptation(dt,Duration,es,V1,a,b);

subplot(212)
plot(t,V);
hold on
[V,impulse_train] = Spike_Rate_Adaptation(dt,Duration,es,V2,a,b);
plot(t,V);
xlim([0 700])
title('Inhabitory Synapses')
%%
clear
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Question2
[I,cmap] = imread('ponyo.jpeg');
[L,Centers] = imsegkmeans(I,4);
B = labeloverlay(I,L);
figure
imshow(B)

%%
clc
clear 
close all

% Q2_K-Mean
I = im2double(imread('ponyo.jpeg')); 
K = 6;
[T, J] = K_Mean(I,K);

figure
imshow(T)
title("by K=Mean,k = "+K)

n = 1:K;

figure
plot(n,J);
title('J(n)');

%%
clc
clear
close all

% Q2_EM
I = im2double(imread('ponyo.jpeg')); 
K = 10;
[maskOut,p,u,v]=kGaussian_color_EM(I,K);
figure
imshow(maskOut)
title("by EM algorithm,k = "+K)
%%
x = (1:225)/225;
y = (1:225)/225;
[X,Y] = meshgrid(x,y);
p1 = reshape(p(:,10),225,[]);
X = [X(:) Y(:)];
%Sigma = [v(10,1),0;0,v(10,1)];
mu = [u(10,1) u(10,2)];
f = mvnpdf(X,mu);
f1 = reshape(f,225,[]);
%%
for i = 1:225
    for j = 1:225
      z(i,j) = p1(i,j)*f1(i,j);
    end
end
contour(x,y,z)
%%

function [V,impulse_train] = Spike_Rate_Adaptation(dti,Durationi,es,Vin,a,b)
dt = dti; 
Duration = Durationi;
T = ceil(Duration/dt);
t = (1:T) * dt;
E_L = -70;
T_m = 20;
R_m = 25;
I_e = 1;
r_m = 100;
Vth = -54;
E_s = es;
V_rest = -80;
T_peak = 100;
V = zeros(T,1);
g = zeros(T,1);
K = zeros(T,1);

V(1) = Vin;
%freq = freqi;

%ro = zeros(T,1);
%th = ceil(1/freq/dt);

K(1) = 0;
for s = 2:T-1
   K(s) = (s-1)/T_peak*exp(1-(s-1)/T_peak); 
end

%a = 0.99995;
%b = 0.00005;
symbols = [0 1];
p = [a b];
impulse_train = randsrc(length(t),1,[symbols;p])';

Q = conv(impulse_train,K);

for i=1:T-1
    g = Q(i);
    V(i+1) = V(i) + dt/T_m*((-V(i)+E_L)-g*(V(i)-E_s)*r_m+I_e*R_m);
    if V(i)>=Vth & V(i+1)>=Vth
        V(i) = 0;
        V(i+1) = V_rest;
    end
end

end

function [T, J] = K_Mean(I,K)
F = reshape(I,size(I,1)*size(I,2),3);                 
                                           
center = F(ceil(rand(K,1)*size(F,1)),:);             
dis = zeros(size(F,1),K+2);                         
                                         
for n = 1:K
   for i = 1:size(F,1)
      for j = 1:K  
        dis(i,j) = norm(F(i,:) - center(j,:));      
      end
      [Distance, CN] = min(dis(i,1:K));               
      dis(i,K+1) = CN;                               
      dis(i,K+2) = Distance;                          
   end
   for i = 1:K
      A = (dis(:,K+1) == i);                      
      center(i,:) = mean(F(A,:));                     
      if sum(isnan(center(:))) ~= 0                    
         NC = find(isnan(center(:,1)) == 1);           
         for Ind = 1:size(NC,1)
         center(NC(Ind),:) = F(randi(size(F,1)),:);
         end
      end
   end
   J(n) = sum(dis,'all');
end
X = zeros(size(F));
for i = 1:K
idx = find(dis(:,K+1) == i);
X(idx,:) = repmat(center(i,:),size(idx,1),1); 
end
T = reshape(X,size(I,1),size(I,2),3);
end


function [maskOut,p,u,v]=kGaussian_color_EM(imageFile,k)
img=double(imageFile);
[M,N,P]=size(img);
n=M*N;
imgR=img(:,:,1); 
imgG=img(:,:,2);
imgB=img(:,:,3);

raw=zeros(n,3);
raw(:,1)=imgR(:);
raw(:,2)=imgG(:);
raw(:,3)=imgB(:);
 
[p,u,v]=em(raw,k);

kColor=u(:,1:3);
imgRe=p*kColor;
maskOut=zeros(M,N,3);
for ii=1:3
    maskOut(:,:,ii)=reshape(imgRe(:,ii),[M,N]);
end

end
 function [p,u,v]=em(raw,k)
[n,dim]=size(raw);
u=raw(randi([1,n],k,1),:);

% initialize standard diviation v
v=zeros(k,1);
for ii=1:k
    raw_tmp=raw(ii:k:end,1);
    v(ii,:)=std(raw_tmp);
end
% initialize weight w
w=ones(k,1)/k;
% initialize membership probability matrix (assignment matrix) p, which is p(k|x)
p=zeros(n,k);
% do interation to get best r
u0=u*0;
v0=0*v;
w0=w*0;
energy=sum(sum((u-u0).^2))+sum(sum((v-v0).^2))+(sum((w-w0).^2));
iteration=1;
x_u=zeros(size(raw));
while energy>10^(-6)
    % calculate membership probability, which is also assignment matrix
    for jj=1:k
        for ss=1:dim
            x_u(:,ss)=raw(:,ss)-u(jj,ss)*ones(n,1);
        end
        x_u=x_u.*x_u;
        p(:,jj)=power(sqrt(2*pi)*v(jj),-1*dim)*exp((-1/2)*sum(x_u,2)./(v(jj).^2));
        p(:,jj)=p(:,jj)*w(jj);
       
    end
    % normalize p on the x dimention
    pSum=sum(p,2);
    for jj=1:k
        p(:,jj)=p(:,jj)./pSum;
    end
    
    % normlaize p on the y dimention, yielding pNorm
    pSum2=sum(p,1);
    pNorm=p*0;
    for jj=1:k
        pNorm(:,jj)=p(:,jj)/pSum2(jj);
    end
    
    % save current u, v, and w as u0, v0 and w0
    u0=u;v0=v;w0=w;
    
    %%update u
    u=(pNorm.')*raw;
    
    % update v
    for jj=1:k
        for ss=1:dim
            x_u(:,ss)=raw(:,ss)-u(jj,ss)*ones(n,1);
        end
        x_u=x_u.*x_u;
        x_uSum=sum(x_u,2);
        v(jj)=sqrt(1/dim*(pNorm(:,jj).')*x_uSum);
    end
    % update w
    w=(sum(p)/n).';
    
    % update and display iteration and energy
    
    %disp(sprintf(['iteration=',num2str(iteration),'; energy=',num2str(energy,'%g')]));
    iteration=iteration+1;
    energy=sum(sum((u-u0).^2))+sum(sum((v-v0).^2))+(sum((w-w0).^2)); 
end
 end
