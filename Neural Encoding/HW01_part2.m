close all
clc
clear

% Homework 01
%% Leaky Integrate and Fire Neuron
% part a

clc
tsim=100; I=20; tau=10; v_r=0; R=1; v_th=15; dt=0.01; % change v_th

step = tsim/dt;
v = zeros(1,step);
t = 0:dt:tsim-dt;
v(1:270) = v_r;

for i = 271:step
    if v(i-1) >= v_th
        v(i-1)= v_th+80;
        v(i)= v_r;
        continue
    end
    v(i) = v(i-1)+ dt*(-v(i-1)+R*I)/tau;
end

figure
plot(t,v);
xlabel('$time (ms)$','Interpreter','latex')
ylabel('$v_{m} (mV)$','Interpreter','latex')
title('Time evolution of membrane potential with zero absolute refactory period','Interpreter','latex')
%% Leaky Integrate and Fire Neuron
% part b

step = tsim/dt; t = 0:dt:tsim-dt; refact = 5;
v = zeros(1,step); v(1:270) = v_r; step = step-271;
s = 271;

while step>=0
    if v(s-1) >= v_th
        v(s-1)= v_th+80;
        v(s:s+refact/dt)= v_r;
        s = s+refact/dt;
        step = step - refact/dt;
        continue
    end
    v(s) = v(s-1)+ dt*(-v(s-1)+R*I)/tau;
    s = s+1; step = step-1;
end

figure
plot(t,v);
xlabel('$time (ms)$','Interpreter','latex')
ylabel('$v_{m} (mV)$','Interpreter','latex')
title('Time evolution of membrane potential with absolute refactory period','Interpreter','latex')

%% Leaky Integrate and Fire Neuron
% part c_ integrator model
clear
clc
%close all

fr=100; tSim=4; k_input=100; dt=1/1000; k=1; tpeak=1.5e-3; tau=13e-3; v_r=0; R=1; v_th=15e-3; a=100;
[t, v, t_spike] = EPSP_integrator(fr,tSim,k_input,dt,k,tpeak,tau,v_r,R,v_th,a);

figure
plot(t,v);
xlim([0 tSim/10])
xlabel('$time (s)$','Interpreter','latex')
ylabel('$v_{m} (mV)$','Interpreter','latex')
title('Neuron Voltage in Time','Interpreter','latex')

%%
% part c_ CV as a function of kernel parameters
fr=100; tSim=1; k_input=100; dt=1/1000; k=1; tpeak=1.5e-3; tau=13e-3; v_r=0; R=1; v_th=15e-3;

% Magnitude
a = 1:150;
cv = [];

for i = a
[~, ~, t_spike] = EPSP_integrator(fr,tSim,k_input,dt,k,tpeak,tau,v_r,R,v_th,i);
ISI = diff(t_spike);
cv = [cv CV_func(ISI)];
end


figure
scatter(a,cv)
xlabel('$ magnitude\ of\ EPSP $','Interpreter','latex')
ylabel('$C_{v}$','Interpreter','latex')
title('$C_{v}$ as a function of magnitude of EPSP','Interpreter','latex')
%%
% width
clc
% Magnitude
fr=100; tSim=1; k_input=100; dt=1/1000; k=1; tau=13e-3; v_r=0; R=1; v_th=15e-3; a=100;
tpeak = 1e-3:0.1e-3:10e-3;
cv = [];

for i = tpeak
[~, ~, t_spike] = EPSP_integrator(fr,tSim,k_input,dt,k,i,tau,v_r,R,v_th,a);
ISI = diff(t_spike);
cv = [cv CV_func(ISI)];
end

figure
scatter(tpeak,cv)
xlabel('$ width\ of\ EPSP (s)$','Interpreter','latex')
ylabel('$C_{v}$','Interpreter','latex')
title('$C_{v}$ as a function of width of EPSP','Interpreter','latex')
%% Leaky Integrate and Fire Neuron
% part d
clear
clc
close all

fr=100; tSim=5; k_input=100; dt=1/1000; k=1; tpeak=1e-3; tau=10e-3; v_r=0; R=1; v_th=15e-4; a=10; r=20;

[t, v, ~] = IPSP_integrator(fr,tSim,k_input,dt,k,tpeak,tau,v_r,R,v_th,a,r);

figure
plot(t,v);
xlim([0 tSim/10])
xlabel('$time (s)$','Interpreter','latex')
ylabel('$v_{m} (mV)$','Interpreter','latex')
title('Neuron Voltage in Time','Interpreter','latex')
%% Leaky Integrate and Fire Neuron
% part d
% CV_Inhibitory percentage
fr=100; tSim=5; k_input=100; dt=1/1000; k=1; tpeak=1e-3; tau=10e-3; v_r=0; R=1; v_th=15e-4; a=30; r=1:40;
cv = [];

for i = r
[t, v, t_spike] = IPSP_integrator(fr,tSim,k_input,dt,k,tpeak,tau,v_r,R,v_th,a,i);
ISI = diff(t_spike);
cv = [cv CV_func(ISI)];
end

figure
scatter(r./k_input.*100,cv);
xlabel('Inhibition\ Percentage$','Interpreter','latex')
ylabel('$C_{v}$','Interpreter','latex')
title('$C_{v}$ as a function of inhibition percentage of synaptic inputs','Interpreter','latex')
%% Leaky Integrate and Fire Neuron
% part e
% Cv as a function of N/M 
clc

fr=100; tSim=2; M=1000; dt=1/1000; k=1; N= 180:2:200; D=2e-3; cv=[];

for i=N
window_num = EPSP_integrator_coincident(fr,tSim,M,i,D,dt,k);
ISI = diff(window_num);
cv = [cv CV_func(ISI)];
end


figure
plot(N./M*100,cv);
ylabel('$C_{v}$','Interpreter','latex')
xlabel('$\frac{N}{M}$','Interpreter','latex')
title('$C_{v}$ as a function of $\frac{N}{M}$','Interpreter','latex')

%% Leaky Integrate and Fire Neuron
% part e
% Cv as a function of D
clc

fr=100; tSim=2; M=200; dt=1/1000; k=1; N= 100;
cv = []; D=2e-3:1e-3:8e-3;

for i = 1:length(D)
window_num = EPSP_integrator_coincident(fr,tSim,M,N,D(i),dt,k);

ISI = diff(window_num);
cv = [cv CV_func(ISI)];

end

figure
plot(D.*1000,cv);
ylabel('$C_{v}$','Interpreter','latex')
xlabel('$D (s)$','Interpreter','latex')
title('$C_{v}$ as a function of D','Interpreter','latex')
%% Leaky Integrate and Fire Neuron
% part f
% cv as a function of Nth
clc

fr=100; tSim=1; M=500; Ni=100:10:220; dt=1/1000; D=2e-3; k=1; Nth=20; cv=[];Nx=M-Ni;

for i=Ni
window_num = IPSP_integrator_coincident(fr,tSim,M,i,Nth,D,dt,k);

ISI = diff(window_num);
cv = [cv CV_func(ISI)];
end

figure
plot(Nx-Ni,cv)
ylabel('$C_{v}$','Interpreter','latex')
xlabel('$N_{x}$','Interpreter','latex')
title('$C_{v}$ as a function of $N{x}$','Interpreter','latex')
%% Leaky Integrate and Fire Neuron
% part f
% cv as a function of D
clc

fr=100; tSim=1; M=400; Ni=100; dt=1/1000; D=3e-3:1e-3:6e-3; k=1; Nth=70; cv=[];Nx=M-Ni;

for i = 1:length(D)
window_num = IPSP_integrator_coincident(fr,tSim,M,Ni,Nth,D(i),dt,k);

ISI = diff(window_num);
cv = [cv CV_func(ISI)];
end

figure
plot(D*1000,cv)
ylabel('$C_{v}$','Interpreter','latex')
xlabel('$D (s)$','Interpreter','latex')
title('$C_{v}$ as a function of D','Interpreter','latex')
%%
function [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials, dt)
nBins = floor(tSim/dt);
spikeMat = rand(nTrials, nBins) < fr*dt;
tVec = 0:dt:tSim-dt;
end

function spikeMat_renewal = renewal_spike_gen(spikeMat, nTrials, k)
spikeMat_renewal = zeros(size(spikeMat));

for j = 1:nTrials
    x = find(spikeMat(j,:));
    for i = k:k:length(x)
        spikeMat_renewal(j,x(i)) = 1;
    end
end
end

function cv = CV_func(x)
cv = std(x)/mean(x);
end

function [t, v, t_spike] = EPSP_integrator(fr,tSim,k_input,dt,k,tpeak,tau,v_r,R,v_th,a)
[spikeMat, t] = poissonSpikeGen(fr, tSim, k_input, dt);
spikeMat_renewal = renewal_spike_gen(spikeMat, k_input, k);

Is = a.*t.*exp(-t./tpeak);
spikeMat_renewal = reshape(spikeMat_renewal,1,[]);
I = conv(spikeMat_renewal,Is,'same');

step = tSim/dt;
v = zeros(1,step);
t_spike = [];

for i = 2:step
    if v(i-1) >= v_th
        v(i-1)= v_th+40e-3;%1 40
        v(i)= v_r;
        t_spike = [t_spike i-1];
        continue
    end
    v(i) = v(i-1)+ dt*(-v(i-1)+R*I(i))/tau;
end
end

function [t, v, t_spike] = IPSP_integrator(fr,tSim,k_input,dt,k,tpeak,tau,v_r,R,v_th,a,r)
[spikeMat, t] = poissonSpikeGen(fr, tSim, k_input, dt);
spikeMat_renewal = renewal_spike_gen(spikeMat, k_input, k);
r_row = randperm(k_input,r);
spikeMat_renewal(r_row,:) = -spikeMat_renewal(r_row,:);
spikeMat_renewal = reshape(spikeMat_renewal,1,[]);
Is = a.*t.*exp(-t./tpeak);
I = conv(spikeMat_renewal,Is,'same');

step = tSim/dt;

v = zeros(1,step); t_spike = [];

for i = 2:step
    if v(i-1) >= v_th
        v(i-1)= v_th+50e-4;
        v(i)= v_r;
        t_spike = [t_spike i-1];
        continue
    end
    v(i) = v(i-1)+ dt*(-v(i-1)+R*I(i))/tau;
end

end

function window_num = EPSP_integrator_coincident(fr,tSim,M,N,D,dt,k)
[spikeMat, ~] = poissonSpikeGen(fr, tSim, M, dt);
spikeMat_renewal = renewal_spike_gen(spikeMat, M, k);

step=D/dt; window_num=[];

for i = 1:size(spikeMat_renewal,2)-step
   if nnz(spikeMat_renewal(:,i:i+step-1)) >= N
       window_num = [window_num i];
   end 
    
end
end



function window_num = IPSP_integrator_coincident(fr,tSim,M,Ni,Nth,D,dt,k)
[spikeMat, ~] = poissonSpikeGen(fr, tSim, M, dt);
spikeMat_renewal = renewal_spike_gen(spikeMat, M, k);

r_row = randperm(M,Ni);
spikeMat_renewal(r_row,:) = -spikeMat_renewal(r_row,:);

step=D/dt; window_num=[];

for i = 1:size(spikeMat_renewal,2)-step
   Np = length(find(spikeMat_renewal(:,i:i+step-1)==1));
   Nn = length(find(spikeMat_renewal(:,i:i+step-1)==-1));
   if Np-Nn >= Nth
       window_num = [window_num i];
   end 
    
end
end