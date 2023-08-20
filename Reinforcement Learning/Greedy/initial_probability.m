function states_probability= initial_probability(x_size,y_size)
states_probability= cell(x_size,y_size);
for i= 1:x_size
    for j= 1:y_size
        if i==1 & j==1
            states_probability{i,j}= [0 0.5 0 0.5]; 
        elseif i==1 & j==y_size
            states_probability{i,j}= [0 0.5 0.5 0]; 
        elseif i==x_size & j==1
            states_probability{i,j}= [0.5 0 0 0.5]; 
        elseif i==x_size & j==y_size
            states_probability{i,j}= [0.5 0 0.5 0]; 
        elseif i==1
         states_probability{i,j}= [1/3 1/3 0 1/3];  
        elseif i==x_size
            states_probability{i,j}= [1/3 1/3 1/3 0]; 
        elseif j==1
            states_probability{i,j}= [1/3 0 1/3 0.25];
        elseif j==y_size
            states_probability{i,j}= [0 1/3 1/3 1/3]; 
        else
        states_probability{i,j}= [0.25 0.25 0.25 0.25]; %above, below, left, right
        end
    end
end
end

