function [policy, m]= policy_update(x_mouse,y_mouse,I,policy,m,learning_rate,state_value,avg_value,state)
beta= 0.1;
if I==1
    m{16-y_mouse,x_mouse}(1)=  m{16-y_mouse,x_mouse}(1)+learning_rate*(1-policy{16-y_mouse,x_mouse}(1))*...
        (state_value(16-(y_mouse+1),x_mouse)-avg_value);
elseif I==2
    m{16-y_mouse,x_mouse}(2)=  m{16-y_mouse,x_mouse}(2)+learning_rate*(1-policy{16-y_mouse,x_mouse}(2))*...
        (state_value(16-(y_mouse-1),x_mouse)-avg_value);
elseif I==3
    m{16-y_mouse,x_mouse}(3)=  m{16-y_mouse,x_mouse}(3)+learning_rate*(1-policy{16-y_mouse,x_mouse}(3))*...
        (state_value(16-(y_mouse),x_mouse-1)-avg_value);
elseif I==4
    m{16-y_mouse,x_mouse}(4)=  m{16-y_mouse,x_mouse}(4)+learning_rate*(1-policy{16-y_mouse,x_mouse}(4))*...
        (state_value(16-(y_mouse),x_mouse+1)-avg_value);
end

for i= 1:length(state)
    if i~= I
        if state(i)==1
            m{16-y_mouse,x_mouse}(1)=  m{16-y_mouse,x_mouse}(1)-learning_rate*(policy{16-y_mouse,x_mouse}(1))*...
                (state_value(16-(y_mouse+1),x_mouse)-avg_value);
        elseif state(i)==2
            m{16-y_mouse,x_mouse}(2)=  m{16-y_mouse,x_mouse}(2)-learning_rate*(policy{16-y_mouse,x_mouse}(2))*...
                (state_value(16-(y_mouse-1),x_mouse)-avg_value);
        elseif state(i)==3
            m{16-y_mouse,x_mouse}(3)=  m{16-y_mouse,x_mouse}(3)-learning_rate*(policy{16-y_mouse,x_mouse}(3))*...
                (state_value(16-(y_mouse),x_mouse-1)-avg_value);
        elseif state(i)==4
            m{16-y_mouse,x_mouse}(4)=  m{16-y_mouse,x_mouse}(4)-learning_rate*(policy{16-y_mouse,x_mouse}(4))*...
                (state_value(16-(y_mouse),x_mouse+1)-avg_value);
        end
    end
end

policy{16-y_mouse,x_mouse}(1)= exp(beta.*m{16-y_mouse,x_mouse}(1))/sum(exp(beta.*m{16-y_mouse,x_mouse}));
policy{16-y_mouse,x_mouse}(2)= exp(beta.*m{16-y_mouse,x_mouse}(2))/sum(exp(beta.*m{16-y_mouse,x_mouse}));
policy{16-y_mouse,x_mouse}(3)= exp(beta.*m{16-y_mouse,x_mouse}(3))/sum(exp(beta.*m{16-y_mouse,x_mouse}));
policy{16-y_mouse,x_mouse}(4)= exp(beta.*m{16-y_mouse,x_mouse}(4))/sum(exp(beta.*m{16-y_mouse,x_mouse}));
end