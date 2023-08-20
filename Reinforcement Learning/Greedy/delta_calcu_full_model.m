function [delta,I, updated_policy, updated_m]= delta_calcu_full_model(x_mouse,y_mouse,action_probability,...
    state_value,m,policy,reward,learning_rate)
if x_mouse==1 & y_mouse==1
%     1
%     delta= policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1);
%     delta
    tmp= [policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse),...
        NaN,NaN,policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1)];
    delta= nansum(tmp);
    [~,idx]= find(tmp==max(tmp));
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse+1),x_mouse)+reward(16-(y_mouse),x_mouse+1))/2;
    direction= [1,4];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif x_mouse==1 & y_mouse==15
%     2
%     delta= policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1);
%     delta
    tmp= [NaN,policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)...
        ,NaN,policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1)];
    delta= nansum(tmp);
    [~,idx]= find(tmp==max(tmp));
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse-1),x_mouse)+reward(16-(y_mouse),x_mouse+1))/2;
    direction= [2,4];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif x_mouse==15 & y_mouse==15
%     3
%     delta= policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1);
%     delta
    tmp= [NaN,policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse),...
        policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1),NaN];
    delta= nansum(tmp);
    [~,idx]= find(tmp==max(tmp));
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse-1),x_mouse)+reward(16-(y_mouse),x_mouse-1))/2;
    direction= [2,3];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif x_mouse==15 & y_mouse==1
%     4
%     delta= policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1);
%     delta
    tmp= [policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse),NaN,...
        policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1),NaN];
    delta= nansum(tmp);
    [~,idx]= find(tmp==max(tmp));
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse+1),x_mouse)+reward(16-(y_mouse),x_mouse-1))/2;
    direction= [1,3];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif y_mouse==1
%     5
%     delta= policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1)+...
%         policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1);
%     delta
    tmp= [policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse),NaN,...
        policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1),...
        policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1)];
    delta= nansum(tmp);
    [~,idx]= find(tmp==max(tmp));
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse+1),x_mouse)+reward(16-(y_mouse),x_mouse-1)+reward(16-(y_mouse),x_mouse+1))/3;
    direction= [1,3,4];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif y_mouse==15
%     6
%     delta= policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1)+...
%         policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1);
%     delta
   
    tmp= [NaN,policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse),...
        policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1),...
        policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1)];
     delta= nansum(tmp);
    [~,idx]= find(tmp==max(tmp));  
    
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse-1),x_mouse)+reward(16-(y_mouse),x_mouse+1)+reward(16-(y_mouse),x_mouse-1))/3;
    direction= [2,3,4];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif x_mouse==15
%     7
%     delta= policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1);
%     delta
    tmp= [policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse),...
        policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse),...
        policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1),NaN];
    [~,idx]= find(tmp==max(tmp));
    delta= nansum(tmp);
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse+1),x_mouse)+reward(16-(y_mouse-1),x_mouse)+reward(16-(y_mouse),x_mouse-1))/3;
    direction= [1,2,3];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
elseif x_mouse==1
%     8
%     delta= policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1);
%     delta
    tmp= [policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse),...
        policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse),NaN,...
        policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1)];
    [~,idx]= find(tmp==max(tmp));
    delta= nansum(tmp);
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse+1),x_mouse)+reward(16-(y_mouse-1),x_mouse)+reward(16-(y_mouse),x_mouse+1))/3;
    direction= [1,2,4];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
    
else
%     9
%     delta= policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse)+...
%         policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1)+...
%         policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1);
    tmp= [policy{16-y_mouse,x_mouse}(1)*action_probability{16-y_mouse,x_mouse}(1)*state_value(16-(y_mouse+1),x_mouse),...
        policy{16-y_mouse,x_mouse}(2)*action_probability{16-y_mouse,x_mouse}(2)*state_value(16-(y_mouse-1),x_mouse),...
        policy{16-y_mouse,x_mouse}(3)*action_probability{16-y_mouse,x_mouse}(3)*state_value(16-(y_mouse),x_mouse-1),...
        policy{16-y_mouse,x_mouse}(4)*action_probability{16-y_mouse,x_mouse}(4)*state_value(16-(y_mouse),x_mouse+1)];
%     delta
    [~,idx]= find(tmp==max(tmp));
    delta= nansum(tmp);
    I= idx(randi([1 length(idx)]));
    avg_reward= (reward(16-(y_mouse+1),x_mouse)+reward(16-(y_mouse-1),x_mouse)+reward(16-(y_mouse),x_mouse+1)+reward(16-(y_mouse),x_mouse-1))/4;
    direction= [1,2,3,4];
    [updated_policy, updated_m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,...
        reward,avg_reward,direction);
end
end