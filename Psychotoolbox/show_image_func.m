function [rects, imageWidth, imageHight, n, deg]= show_image_func(wPtr,mat,img_num,num,CenterX,CenterY,good_label,zero_angle)
radius= 400; deg= (360/img_num:360/img_num:360).*pi/180+zero_angle; T=[]; rects=[];

% check if the angle is bigger than 360
for i=1:length(deg)
   if deg(i) >= 360
       deg(i)=deg(i)-360;
   end
end

for i=1:img_num
x= CenterX+radius*cos(deg(i)); y= CenterY+radius*sin(deg(i));
img= squeeze(mat(:,:,:,num(i)));
faceTexture = Screen('MakeTexture',wPtr,img); 
T= [T faceTexture];
[imageHight, imageWidth, ~]= size(img);

n=5;
rect= CenterRectOnPointd([0 0 imageWidth/n imageHight/n], x, y)';
rects= [rects rect];

if isempty(find(good_label==num(i), 1))
   color= [255 0 0];
else
    color= [0 255 0];
end
Screen('FrameRect',wPtr,color,[x-imageWidth/n+50 y-imageHight/n+50 x+imageWidth/n-50 y+imageHight/n-50]);
end
Screen('DrawTextures',wPtr,T,[],rects);
end