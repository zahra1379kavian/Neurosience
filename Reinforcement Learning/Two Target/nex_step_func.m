function I= nex_step_func(tmp)
max_value= max(tmp,[],'all');
idx= [];

for i= 1:length(tmp)
    if ~isnan(tmp(i)) && tmp(i)>= max_value-0.01
        idx= [idx, i];
    end
end
I= idx(randi([1 length(idx)]));
end