function [X1, X2, Choice]= extend_race_trial(positive_thr, negative_thr, sigma1, sigma2, X01, X02, bias1, bias2, time_interval)
mu= 0; dt= 0.1; Choice= 0;
t= 0:dt:time_interval-dt;
W1= zeros(size(t));
W2= zeros(size(t));
W1(2:end)= normrnd(mu,sqrt(dt),[1 length(t)-1]);
W2(2:end)= normrnd(mu,sqrt(dt),[1 length(t)-1]);
X1(1)= X01; X2(1)= X02;

for i= 2:length(t)  
    X1(i)= X1(i-1)+ bias1*dt+ sigma1* W1(i);
    X2(i)= X2(i-1)+ bias2*dt+ sigma2* W2(i);
    
    if X1(i)>= positive_thr
        Choice= 1;
        X1(i)= positive_thr;
        break   
    elseif X2(i)<= negative_thr
        Choice= -1;
        X2(i)= negative_thr;
        break
    end
end

if Choice==0
    indx= randi(2);
    if indx==1
        Choice= 1; 
        X1(end)= positive_thr;
    else
        Choice= -1;
        X2(end)= negative_thr;
    end
end
end