clear
clc
close all

%% Hodgkin-Huxley Model
%Q1.1,1.2,1.3,1.4,1.8 

%Iin min
%Iin = 3;
%Time min
%tt = 8;
Iin = 2000;
tt = 10;
%Domain = SteadtState_TimeCons(Iin);
SteadtState_TimeCons(Iin,tt);

%%
clear
clc
close all

%Q1.5
tt = 10;

Iin = zeros(10,1);
Di = zeros(10,1);
Pi = zeros(10,1);

for i = 1:1:10
Iin(i) = 3.5+i;
[D(i) P(i)] = Domain_Phase_Plot(Iin(i),tt);
end

figure
plot(Iin,D);
grid minor
title('Domain Spike');

figure
plot(Iin,P);
grid minor
title('Frequenc Spike');

%%
%clear
clc
close all

%Q1.7

tt = 15;
part_seven_plot(tt);

%%
%clear
clc
close all

%Q1.9
a = 20;
tt = 15;
dt = 0.01; % Simulation time step
Duration = 50; % Simulation length
T = ceil(Duration/dt);

I = a*triang(T);

DiffrentInput(tt,I);

%%
clear
clc
close all

%Q1.9
tt = 15;
dt = 0.01; % Simulation time step
Duration = 50; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt;

rect=@(x,a) ones(1,numel(x)).*(abs(x)<a/2);% a is the width of the pulse
I=20*rect(t,100)';

DiffrentInput(tt,I);


%%
clear
clc
close all

%Q1.9
tt = 15;
dt = 0.01; % Simulation time step
Duration = 200; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt;

I = chirp(t,0,1500,100,'quadratic');

DiffrentInput(tt,I);

%%
clear
clc
close all

%Q1.9
tt = 15;
dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt;
f = 0.05;
w = 2*pi*f;

I = 20*sin(w*t);

DiffrentInput(tt,I);

%%
clear
clc
close all

%Q1.10

tt = 20;

for i = 1:20
    I(i) = 4+i;
    V(i) = F_ICurve_HH(tt,I(i));
end

plot(I,V);
grid minor
title('F-I curve');
xlim([7 24]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Izhikevich Model
clear
clc
close all

%Q2

%Tonic Spiking

ai = 0.02;
bi = 0.2;
ci = -65;
di = 2;
hi = 15;
ti = 100;
Izhikevich(ai,bi,ci,di,hi,ti);
%%
clear
clc
close all

%Phasic Spiking

ai = 0.02;
bi = 0.25;
ci = -65;
di = 6;
hi = 1;
ti = 120;
Izhikevich(ai,bi,ci,di,hi,ti);
U_VIzhikevich(ai,bi,ci,di,hi,ti);

%%
clear
clc
close all

%Tonic Bursting

ai = 0.02;
bi = 0.2;
ci = -50;
di = 2;
hi = 15;
ti = 100;
Izhikevich(ai,bi,ci,di,hi,ti);

%%
clear
clc
close all

%Phasic Bursting 

ai = 0.02;
bi = 0.25;
ci = -55;
di = 0.05;
hi = 0.7;
ti = 250;
Izhikevich(ai,bi,ci,di,hi,ti);

%%
clear
clc
close all

%Mixed Model 

ai = 0.02;
bi = 0.2;
ci = -55;
di = 4;
hi = 10;
ti = 250;
Izhikevich(ai,bi,ci,di,hi,ti);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Noisy Output Model
clear
clc
close all

%Q3

tt = 20;
Iin = 2;

LIF_Module(tt,Iin);

%%
clear
clc
close all

%Q3
tt = 20;

for i = 1:500
    I(i) = 20+i;
    V(i) = F_ICurve_LIF(tt,I(i));
end

plot(I,V);
grid minor
title('F-I curve');

%%
function SteadtState_TimeCons(Iin,tt)
dt = 0.01; % Simulation time step
Duration = 20000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
Cm = 1; % Membrane capacitance in micro Farads
gNa = 120; % in Siemens, maximum conductivity of Na+ Channel
gK = 36; % in Siemens, maximum conductivity of K+ Channel
gl = 0.3; % in Siemens, conductivity of leak Channel
ENa = 55; % in mv, Na+ nernst potential
EK = -72; % in mv, K+ nernst potential
El = -49.4; % in mv, nernst potential for leak channel
%vRest = -60; % in mv, resting potential
%V = vRest * ones(T,1); % Vector of output voltage
I = zeros(T,1); % in uA, external stimulus (external current)
% for example: I(1:10000) = 2; % an input current pulse


V = zeros(T,1);
m = zeros(T,1);
h = zeros(T,1);
n = zeros(T,1);
alpha_m = zeros(T,1);
alpha_n = zeros(T,1);
alpha_h = zeros(T,1);
beta_m = zeros(T,1);
beta_n = zeros(T,1);
beta_h = zeros(T,1);

m(1)=0.5;
h(1)=0.02;
n(1)=0.65;
I(T,1) = Iin;
vRest = -60;
V(1) = vRest;

for i=1:T-1    
    alpha_m(i) = alphaM(V(i));
    beta_m(i) = betaM(V(i));
    m(i+1) = m(i)+dt*((alpha_m(i)+beta_m(i))*(-m(i)+alpha_m(i)/(alpha_m(i)+beta_m(i))));
    
    alpha_h(i) = alphaH(V(i));
    beta_h(i) = betaH(V(i));
    h(i+1) = h(i)+dt*((alpha_h(i)+beta_h(i))*(-h(i)+alpha_h(i)/(alpha_h(i)+beta_h(i))));
    
    alpha_n(i) = alphaN(V(i));
    beta_n(i) = betaN(V(i));
    n(i+1) = n(i)+dt*((alpha_n(i)+beta_n(i))*(-n(i)+alpha_n(i)/(alpha_n(i)+beta_n(i))));
    
    V(i+1) = V(i) + dt*(gNa*m(i)^3*h(i)*(ENa-V(i)) + gK*n(i)^4*(EK-V(i)) + gl*(El-V(i)) + I(i));
    
    if i>=tt/dt
        I(i+1) = Iin;
    else
        I(i+1) = 0;
    end
end


n_inf = alpha_n./(alpha_n+beta_n);
m_inf = alpha_m./(alpha_m+beta_m);
h_inf = alpha_h./(alpha_h+beta_h);

T_h = 1./(alpha_h+beta_h);
T_m = 1./(alpha_m+beta_m);
T_n = 1./(alpha_n+beta_n);

figure
plot(V,h_inf);
hold on
plot(V,m_inf);
hold on
plot(V,n_inf);
grid minor
legend('h_{inf}','m_{inf}','n_{inf}')
title('Steady State Value');
xlim([-70 10])

figure
plot(V,T_h);
hold on
plot(V,T_m);
hold on
plot(V,T_n);
grid minor
legend('T_{h}','T_{m}','T_{n}')
title('Time Constant');
xlim([-70 10])

figure
plot(t,V);
grid minor
title('Cell Membrane Voltage');
xlim([-5 50])

figure
plot(V,n);
grid minor
title('V-n function');

figure
plot(V,m);
grid minor
title('V-m function');

figure
plot(V,h);
grid minor
title('V-h function');
end

function [Domain, meanCycle] = Domain_Phase_Plot(Iin,tt)
dt = 0.01; % Simulation time step
Duration = 20000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
Cm = 1; % Membrane capacitance in micro Farads
gNa = 120; % in Siemens, maximum conductivity of Na+ Channel
gK = 36; % in Siemens, maximum conductivity of K+ Channel
gl = 0.3; % in Siemens, conductivity of leak Channel
ENa = 55; % in mv, Na+ nernst potential
EK = -72; % in mv, K+ nernst potential
El = -49.4; % in mv, nernst potential for leak channel
%vRest = -60; % in mv, resting potential
%V = vRest * ones(T,1); % Vector of output voltage
I = zeros(T,1); % in uA, external stimulus (external current)
% for example: I(1:10000) = 2; % an input current pulse


V = zeros(T,1);
m = zeros(T,1);
h = zeros(T,1);
n = zeros(T,1);
alpha_m = zeros(T,1);
alpha_n = zeros(T,1);
alpha_h = zeros(T,1);
beta_m = zeros(T,1);
beta_n = zeros(T,1);
beta_h = zeros(T,1);

m(1)=0.5;
h(1)=0.02;
n(1)=0.65;
I(T,1) = Iin;
vRest = -60;
V(1) = vRest;

for i=1:T-1    
    alpha_m(i) = alphaM(V(i));
    beta_m(i) = betaM(V(i));
    m(i+1) = m(i)+dt*((alpha_m(i)+beta_m(i))*(-m(i)+alpha_m(i)/(alpha_m(i)+beta_m(i))));
    
    alpha_h(i) = alphaH(V(i));
    beta_h(i) = betaH(V(i));
    h(i+1) = h(i)+dt*((alpha_h(i)+beta_h(i))*(-h(i)+alpha_h(i)/(alpha_h(i)+beta_h(i))));
    
    alpha_n(i) = alphaN(V(i));
    beta_n(i) = betaN(V(i));
    n(i+1) = n(i)+dt*((alpha_n(i)+beta_n(i))*(-n(i)+alpha_n(i)/(alpha_n(i)+beta_n(i))));
    
    V(i+1) = V(i) + dt*(gNa*m(i)^3*h(i)*(ENa-V(i)) + gK*n(i)^4*(EK-V(i)) + gl*(El-V(i)) + I(i));
    
    if i>=tt/dt
        I(i+1) = Iin;
    else
        I(i+1) = 0;
    end
end

[pks,locs] = findpeaks(V,t);
meanCycle = mean(diff(locs));
Domain = max(pks);
end

function Vr = F_ICurve_HH(tt,Iin)
dt = 0.01; % Simulation time step
Duration = 20000; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
Cm = 1; % Membrane capacitance in micro Farads
gNa = 120; % in Siemens, maximum conductivity of Na+ Channel
gK = 36; % in Siemens, maximum conductivity of K+ Channel
gl = 0.3; % in Siemens, conductivity of leak Channel
ENa = 55; % in mv, Na+ nernst potential
EK = -72; % in mv, K+ nernst potential
El = -49.4; % in mv, nernst potential for leak channel
%vRest = -60; % in mv, resting potential
%V = vRest * ones(T,1); % Vector of output voltage
I = zeros(T,1); % in uA, external stimulus (external current)
% for example: I(1:10000) = 2; % an input current pulse


V = zeros(T,1);
m = zeros(T,1);
h = zeros(T,1);
n = zeros(T,1);
alpha_m = zeros(T,1);
alpha_n = zeros(T,1);
alpha_h = zeros(T,1);
beta_m = zeros(T,1);
beta_n = zeros(T,1);
beta_h = zeros(T,1);

m(1)=0.5;
h(1)=0.02;
n(1)=0.65;
I(T,1) = Iin;
vRest = -60;
V(1) = vRest;
Vth = -45;

count2 = 0;
count3 = 1;

for i=1:T-1    
    alpha_m(i) = alphaM(V(i));
    beta_m(i) = betaM(V(i));
    m(i+1) = m(i)+dt*((alpha_m(i)+beta_m(i))*(-m(i)+alpha_m(i)/(alpha_m(i)+beta_m(i))));
    
    alpha_h(i) = alphaH(V(i));
    beta_h(i) = betaH(V(i));
    h(i+1) = h(i)+dt*((alpha_h(i)+beta_h(i))*(-h(i)+alpha_h(i)/(alpha_h(i)+beta_h(i))));
    
    alpha_n(i) = alphaN(V(i));
    beta_n(i) = betaN(V(i));
    n(i+1) = n(i)+dt*((alpha_n(i)+beta_n(i))*(-n(i)+alpha_n(i)/(alpha_n(i)+beta_n(i))));
    
    V(i+1) = V(i) + dt*(gNa*m(i)^3*h(i)*(ENa-V(i)) + gK*n(i)^4*(EK-V(i)) + gl*(El-V(i)) + I(i));
    
    if V(i+1)>=Vth
        count2 = count2 +1;
    end
    
    if i>=tt/dt
        I(i+1) = Iin;
    else
        I(i+1) = 0;
    end
    
    if mod(i,1/dt) == 0
        Vrate(count3) = count2;
        count2 = 0;
        count3 = count3+1;
    end
end
Vr = mean(Vrate);
end

function part_seven_plot(tt)
dt = 0.01; % Simulation time step
Duration = 50; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
Cm = 1; % Membrane capacitance in micro Farads
gNa = 120; % in Siemens, maximum conductivity of Na+ Channel
gK = 36; % in Siemens, maximum conductivity of K+ Channel
gl = 0.3; % in Siemens, conductivity of leak Channel
ENa = 55; % in mv, Na+ nernst potential
EK = -72; % in mv, K+ nernst potential
El = -49.4; % in mv, nernst potential for leak channel
%vRest = -60; % in mv, resting potential
%V = vRest * ones(T,1); % Vector of output voltage
%I = zeros(T,1); % in uA, external stimulus (external current)
% for example: I(1:10000) = 2; % an input current pulse


V = zeros(T,1);
m = zeros(T,1);
h = zeros(T,1);
n = zeros(T,1);
I = zeros(T,1);

alpha_m = zeros(T,1);
alpha_n = zeros(T,1);
alpha_h = zeros(T,1);
beta_m = zeros(T,1);
beta_n = zeros(T,1);
beta_h = zeros(T,1);

m(1)=0.5;
h(1)=0.02;
n(1)=0.65;
vRest = -60;
V(1) = vRest;
I(1) = 1;

for i=1:T-1    
    alpha_m(i) = alphaM(V(i));
    beta_m(i) = betaM(V(i));
    m(i+1) = m(i)+dt*((alpha_m(i)+beta_m(i))*(-m(i)+alpha_m(i)/(alpha_m(i)+beta_m(i))));
    
    alpha_h(i) = alphaH(V(i));
    beta_h(i) = betaH(V(i));
    h(i+1) = h(i)+dt*((alpha_h(i)+beta_h(i))*(-h(i)+alpha_h(i)/(alpha_h(i)+beta_h(i))));
    
    alpha_n(i) = alphaN(V(i));
    beta_n(i) = betaN(V(i));
    n(i+1) = n(i)+dt*((alpha_n(i)+beta_n(i))*(-n(i)+alpha_n(i)/(alpha_n(i)+beta_n(i))));
    
    V(i+1) = V(i) + dt*(gNa*m(i)^3*h(i)*(ENa-V(i)) + gK*n(i)^4*(EK-V(i)) + gl*(El-V(i)) + I(i));
    
    if i>=tt/dt
        I(i+1) = i+1;
    else
        I(i+1) = 0;
    end
end

figure
plot(t,V);
grid minor
title('Cell Membrane Voltage');
%xlim([-5 50])
end

function DiffrentInput(tt,I)
dt = 0.01; % Simulation time step
Duration = 100; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms
Cm = 1; % Membrane capacitance in micro Farads
gNa = 120; % in Siemens, maximum conductivity of Na+ Channel
gK = 36; % in Siemens, maximum conductivity of K+ Channel
gl = 0.3; % in Siemens, conductivity of leak Channel
ENa = 55; % in mv, Na+ nernst potential
EK = -72; % in mv, K+ nernst potential
El = -49.4; % in mv, nernst potential for leak channel
%vRest = -60; % in mv, resting potential
%V = vRest * ones(T,1); % Vector of output voltage
%I = zeros(T,1); % in uA, external stimulus (external current)
% for example: I(1:10000) = 2; % an input current pulse


V = zeros(T,1);
m = zeros(T,1);
h = zeros(T,1);
n = zeros(T,1);

%I = a*triang(T);
%I = zeros(T,1);

alpha_m = zeros(T,1);
alpha_n = zeros(T,1);
alpha_h = zeros(T,1);
beta_m = zeros(T,1);
beta_n = zeros(T,1);
beta_h = zeros(T,1);

m(1)=0.5;
h(1)=0.02;
n(1)=0.65;
vRest = -60;
V(1) = vRest;
%I(1) = 1;

for i=1:T-1    
    alpha_m(i) = alphaM(V(i));
    beta_m(i) = betaM(V(i));
    m(i+1) = m(i)+dt*((alpha_m(i)+beta_m(i))*(-m(i)+alpha_m(i)/(alpha_m(i)+beta_m(i))));
    
    alpha_h(i) = alphaH(V(i));
    beta_h(i) = betaH(V(i));
    h(i+1) = h(i)+dt*((alpha_h(i)+beta_h(i))*(-h(i)+alpha_h(i)/(alpha_h(i)+beta_h(i))));
    
    alpha_n(i) = alphaN(V(i));
    beta_n(i) = betaN(V(i));
    n(i+1) = n(i)+dt*((alpha_n(i)+beta_n(i))*(-n(i)+alpha_n(i)/(alpha_n(i)+beta_n(i))));
    
    V(i+1) = V(i) + dt*(gNa*m(i)^3*h(i)*(ENa-V(i)) + gK*n(i)^4*(EK-V(i)) + gl*(El-V(i)) + I(i));
    
    if i<tt/dt
        I(i+1) = 0;
    end
end

figure
plot(t,V);
grid minor
title('Cell Membrane Voltage');
xlim([10 80])
end

function alpha_m = alphaM(v)
vRest = -60;
u = vRest - v;
alpha_m = (u+25) ./ (exp(2.5+.1*u)-1)/10;
end

function beta_m = betaM(v)
vRest = -60;
u = vRest - v;
beta_m = 4*exp(u/18);
end

function alpha_h = alphaH(v)
vRest = -60;
u = vRest - v;
alpha_h = .07 * exp(u/20);
end

function beta_h = betaH(v)
vRest = -60;
u = vRest - v;
beta_h = 1./(1+exp(3+.1*u));
end

function alpha_n = alphaN(v)
vRest = -60;
u = vRest - v;
alpha_n = (.1 * u + 1)./(exp(1 + .1 * u) - 1) / 10;
end

function beta_n = betaN(v)
vRest = -60;
u = vRest - v;
beta_n = .125 * exp(u/80);
end

function Izhikevich(ai,bi,ci,di,hi,ti)
dt = 0.1; % Simulation time step
Duration = ti; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms

a = ai;
b = bi;
c = ci;
d = di;
h = hi;

%V = -65.*ones(T,1);
V = zeros(T,1);
%U = b.*V;
U = zeros(T,1);
I = zeros(T,1);
V(1,1) = c;
U(1,1) = c*b;

for i = 1:T-1
    fired=find(V>=30);
    V(fired)=c;
    U(fired)=U(fired)+d;
    
    V(i+1) = V(i) + dt.*(0.04.*V(i)^2+5.*V(i)+140-U(i)+I(i));
    U(i+1) = U(i) + dt.*(a.*(b.*V(i)-U(i)));
    if i>=10/dt
    I(i+1) = h;
    else
        I(i+1) = 0;
    end
end

figure
plot(t,V);
grid minor
title('Cell Membrane Voltage');

figure
plot(t,I);
grid minor
title('Input Current')
xlim([0 Duration])
ylim([0 h+2])
end

function U_VIzhikevich(ai,bi,ci,di,hi,ti)
dt = 0.1; % Simulation time step
Duration = ti; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms

a = ai;
b = bi;
c = ci;
d = di;
h = hi;

%V = -65.*ones(T,1);
V = zeros(T,1);
%U = b.*V;
U = zeros(T,1);
I = zeros(T,1);
V(1,1) = c;
U(1,1) = c*b;

for i = 1:T-1
    fired=find(V>=30);
    V(fired)=c;
    U(fired)=U(fired)+d;
    
    V(i+1) = V(i) + dt.*(0.04.*V(i)^2+5.*V(i)+140-U(i)+I(i));
    U(i+1) = U(i) + dt.*(a.*(b.*V(i)-U(i)));
    if i>=10/dt
    I(i+1) = h;
    else
        I(i+1) = 0;
    end
end

figure
plot(V,U);
grid minor
title('V-U function pattern spiking phasic ');
end

function LIF_Module(tt,Iin)
dt = 0.001; % Simulation time step
Duration = 200; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms

V = zeros(T,1);
I = zeros(T,1);

R = 10;
Vth = -45;
Vrest = -70;
Vpeak = 15;
Tm = 10;
gama = 0.001;
beta = 2;
count1 = 1;

V(1,1) = Vrest;
I(1,1) = Iin;

for i = 1:T-1
    
    x = V(i)-Vth;
    Vm = beta*(1+tanh(gama*x));
    
    V(i+1) = V(i) + dt*(-V(i)+R*I(i));
    
    if V(i+1)>=Vpeak
        V(i+1) = Vrest;
    elseif V(i+1)>=Vm
        Vs(count1) = V(i+1);
        count1 = count1 +1;
        V(i+1) = Vpeak;
    end
    
    if i>=tt/dt
        I(i+1) = Iin;
    else
        I(i+1) = 0;
    end
end

figure
plot(t,V);
grid minor
title('Cell Membrane Voltage');
xlim([0 40])

figure 
h = histogram(Vs)
end

function Vr = F_ICurve_LIF(tt,Iin)
dt = 0.01; % Simulation time step
Duration = 200; % Simulation length
T = ceil(Duration/dt);
t = (1:T) * dt; % Simulation time points in ms

V = zeros(T,1);
I = zeros(T,1);

R = 1;
Vth = -45;
Vrest = -70;
Vpeak = 15;
Tm = 10;
gama = 0.001;
beta = 2;
count1 = 1;
count2 = 0;
count3 = 1;

V(1,1) = Vrest;
I(1,1) = Iin;

for i = 1:T-1
    
    x = V(i)-Vth;
    Vm = beta*(1+tanh(gama*x));
    
    V(i+1) = V(i) + dt*(-V(i)+R*I(i));
    
    if V(i+1)>=Vpeak
        V(i+1) = Vrest;
    elseif V(i+1)>=Vm
        Vs(count1) = V(i+1);
        count1 = count1 +1;
        count2 = count2 +1;
        V(i+1) = Vpeak;
    end
    
    if i>=tt/dt
        I(i+1) = Iin;
    else
        I(i+1) = 0;
    end
    
    if mod(i,1/dt) == 0
        Vrate(count3) = count2;
        count2 = 0;
        count3 = count3+1;
    end
end

Vr = mean(Vrate);
end