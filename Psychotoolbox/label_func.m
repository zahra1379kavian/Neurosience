function [good_label, bad_label]= label_func()
image_num= 48; s=zeros(1,image_num);
good_label= randperm(48,image_num/2); s(good_label)=1;
bad_label=find(~s);
end
