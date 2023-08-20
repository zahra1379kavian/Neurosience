function [X1, X2, RT, Choice]= race_trial(positive_thr, negative_thr, sigma1, sigma2, X01, X02, bias1, bias2)
X1(1)= X01;    X2(1)= X02;
mu= 0; dt= 0.1;
count= 1;
flag= 0;

% while X1(count)>negative_thr & X1(count)<positive_thr &  X2(count)>negative_thr & X2(count)<positive_thr
while ~flag
    W1= normrnd(mu,sqrt(dt));
    W2= normrnd(mu,sqrt(dt));
    
    X1(count+1)= X1(count)+ bias1*dt+ sigma1* W1;
    X2(count+1)= X2(count)+ bias2*dt+ sigma2* W2;
    count= count+1;
    
    if X1(count)<= negative_thr
        RT= count*dt;
        Choice= -1;
        X1(count)= negative_thr;
        flag= 1;
        break
    elseif X1(count)>= positive_thr
        RT= count*dt;
        Choice= 1;
        X1(count)= positive_thr;
        flag= 1;
        break
    end

    if X2(count)<= negative_thr
        RT= count*dt;
        Choice= -1;
        X2(count)= negative_thr;
        flag= 1;
        break
    elseif X2(count)>= positive_thr
        RT= count*dt;
        Choice= 1;
        X2(count)= positive_thr;
        flag= 1;
        break
    end
end


end