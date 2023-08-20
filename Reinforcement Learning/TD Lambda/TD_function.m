function state_value= TD_function(old_state,state_value,x_mouse,y_mouse,lambda,iter_number)
delta= state_value(16-y_mouse(iter_number),x_mouse(iter_number))-old_state(16-y_mouse(iter_number),x_mouse(iter_number));

for i= 1:iter_number-1
    state_value(16-y_mouse(i),x_mouse(i))= old_state(16-y_mouse(i),x_mouse(i))+...
      lambda^(iter_number-i)*(delta);  
  
end
end