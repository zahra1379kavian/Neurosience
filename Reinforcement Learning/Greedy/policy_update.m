function [policy, m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,action_probability,reward,avg_reward,state)
beta= 1;
if I==1
    m{16-y_mouse,x_mouse}(1)=  m{16-y_mouse,x_mouse}(1)+learning_rate*(1-action_probability{16-y_mouse,x_mouse}(1))*...
        (reward(16-(y_mouse+1),x_mouse)-avg_reward);
elseif I==2
    m{16-y_mouse,x_mouse}(2)=  m{16-y_mouse,x_mouse}(2)+learning_rate*(1-action_probability{16-y_mouse,x_mouse}(2))*...
        (reward(16-(y_mouse-1),x_mouse)-avg_reward);
elseif I==3
    m{16-y_mouse,x_mouse}(3)=  m{16-y_mouse,x_mouse}(3)+learning_rate*(1-action_probability{16-y_mouse,x_mouse}(3))*...
        (reward(16-(y_mouse),x_mouse-1)-avg_reward);
elseif I==4
    m{16-y_mouse,x_mouse}(4)=  m{16-y_mouse,x_mouse}(4)+learning_rate*(1-action_probability{16-y_mouse,x_mouse}(4))*...
        (reward(16-(y_mouse),x_mouse+1)-avg_reward);
end
for i= 1:length(state)
    if i~= I
        if state(i)==1
            m{16-y_mouse,x_mouse}(1)=  m{16-y_mouse,x_mouse}(1)+learning_rate*(action_probability{16-y_mouse,x_mouse}(1))*...
                (reward(16-(y_mouse+1),x_mouse)-avg_reward);
        elseif state(i)==2
            m{16-y_mouse,x_mouse}(2)=  m{16-y_mouse,x_mouse}(2)+learning_rate*(action_probability{16-y_mouse,x_mouse}(2))*...
                (reward(16-(y_mouse-1),x_mouse)-avg_reward);
        elseif state(i)==3
            m{16-y_mouse,x_mouse}(3)=  m{16-y_mouse,x_mouse}(3)+learning_rate*(action_probability{16-y_mouse,x_mouse}(3))*...
                (reward(16-(y_mouse),x_mouse-1)-avg_reward);
        elseif state(i)==4
            m{16-y_mouse,x_mouse}(4)=  m{16-y_mouse,x_mouse}(4)+learning_rate*(action_probability{16-y_mouse,x_mouse}(4))*...
                (reward(16-(y_mouse),x_mouse+1)-avg_reward);
        end
    end
end
%I
%exp(beta*m{16-y_mouse,x_mouse}(I))/sum(exp(m{16-y_mouse,x_mouse}))
policy{16-y_mouse,x_mouse}(1)= exp(beta.*m{16-y_mouse,x_mouse}(1))./sum(exp(m{16-y_mouse,x_mouse}));
policy{16-y_mouse,x_mouse}(2)= exp(beta.*m{16-y_mouse,x_mouse}(2))./sum(exp(m{16-y_mouse,x_mouse}));
policy{16-y_mouse,x_mouse}(3)= exp(beta.*m{16-y_mouse,x_mouse}(3))./sum(exp(m{16-y_mouse,x_mouse}));
policy{16-y_mouse,x_mouse}(4)= exp(beta.*m{16-y_mouse,x_mouse}(4))./sum(exp(m{16-y_mouse,x_mouse}));
end