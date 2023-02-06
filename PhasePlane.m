
I = 0.3;
phaseplane(I);
%%
I = 0:1:5;
for i = 1:size(I,2)
  phaseplane(I(i));
  pause(1)
end  



%%
function phaseplane(Iin)
figure
% initial parameters
noip = 15;
interval = 5;
I = Iin;
f = @(t,Y) [Y(1)-Y(1)^3-Y(2)+I; 0.08*(Y(1)+.7-.8*Y(2))];
y1 = linspace(-interval,interval,20);
y2 = linspace(-interval,interval,20);
% creates two matrices one for all the x-values on the grid, and one for
% all the y-values on the grid. Note that x and y are matrices of the same
% size and shape, in this case 20 rows and 20 columns
[x,y] = meshgrid(y1,y2);
u = zeros(size(x));
v = zeros(size(x));
% we can use a single loop over each element to compute the derivatives at
% each point (y1, y2)
t=0; % we want the derivatives at each point at t=0, i.e. the starting time
for i = 1:numel(x)
Yprime = f(t,[x(i); y(i)]);
u(i) = Yprime(1);
v(i) = Yprime(2);
end
quiver(x,y,u,v,'color','#8c8c8c');
xlabel('v','Interpreter','latex')
ylabel('w','Interpreter','latex')
% axis tight equal;
hold on
for i = 1:noip
[ts,ys] = ode45(f,[0,100],[rand()*interval*((-1)^floor(rand()*interval)); ...
rand()*interval*((-1)^floor(rand()*interval))]);
plot(ys(:,1),ys(:,2),'color','#5ea6ed','LineWidth',1.5)
plot(ys(1,1),ys(1,2),'bo') % starting point
plot(ys(end,1),ys(end,2),'ks') % ending point
xlim([-interval interval]);
ylim([-interval interval]);
end

[ts,ys] = ode45(f,[0,100],[0.5,1]);
plot(ys(:,1),ys(:,2),'color','#5ea6ed','LineWidth',1.5)
plot(ys(1,1),ys(1,2),'bo') % starting point
plot(ys(end,1),ys(end,2),'ks') % ending point
xlim([-interval interval]);
ylim([-interval interval]);

syms v
fplot(v-v^3+I,'color','#800000','LineWidth',1.5);
fplot((v+.7)/.8,'color','#ff8c1a','LineWidth',1.5);
title("$Phase\;Plane, I=$"+I,'Interpreter','latex')
hold('off')
end