function [X, RT, Choice]= two_choice_trial(positive_thr, negative_thr,sigma, X0, bias_pos,bias_neg)
X(1)= X0;
mu= 0; dt= 0.1;
count= 1;

while X(count)>negative_thr && X(count)<positive_thr
    W= normrnd(mu,sqrt(dt));
    if X(count)>0
        X(count+1)= X(count)+ bias_pos*dt+ sigma* W;
    else
        X(count+1)= X(count)+ bias_neg*dt+ sigma* W;
    end
    count= count+1;
    if X(count)<= negative_thr
        RT= count*dt;
        Choice= -1;
        X(count)= negative_thr;
        break
    elseif X(count)>= positive_thr
        RT= count*dt;
        Choice= 1;
        X(count)= positive_thr;
        break
    end
end


end