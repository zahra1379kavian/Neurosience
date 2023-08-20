function [X, Choice]= simple_model(bias, sigma, mu, dt, time_interval)

t= 0:dt:time_interval-dt; 
W= zeros(size(t)); X= zeros(size(t)); 
W(2:end)= normrnd(mu,sqrt(dt),[1 length(t)-1]);

for i= 2:length(t)
   X(i)= X(i-1)+ bias*dt+ sigma* W(i);
end

if X(end)>0; Choice=1; else; Choice= 0; end

end