function [LIP_event_times,count,MT_spike,LIP_event,p_lip,dt]= LIP_activity_part2(MT_p_values,LIP_weights,LIP_threshold,Evidence_thr)
% Parameters:
% MT_p_values - a vector with 2 elements, firing probabilities for the
% excitatory and inhibitory neurons, resp.
% LIP_weights - a length 2 vector of weighting factors for the evidence
% from the excitatory (positive) and
% inhibitory (negative) neurons
% LIP_threshold - the LIP firing rate that represents the choice threshold criterion
% use fixed time scale of 1 ms

dt= 0.01;
rate= 0;
N= [0 0]; % plus is first, minus is second
count= 1;
LIP_event_times= [];

while rate<LIP_threshold
   dN= rand(1,2) < MT_p_values;
   N= N+ dN;
   
   MT_spike(count,:)= dN;
   
   p_lip(count)= sum(N.*LIP_weights);
   
   LIP_event(count)= p_lip(count)> Evidence_thr;
   
   if LIP_event(count) == 1
       LIP_event_times = [LIP_event_times t];
   end
   
   % check LIP mean rate for last M spikes
   M = 100;
   if length(LIP_event_times)>=M
       rate = M/(t-LIP_event_times(length(LIP_event_times)-M+1));
   end
   count= count+1;
   t= (count-1)*dt;
end
end
