function reward_show_func(wPtr, reward, reply, CenterX, CenterY, key)
% open file
fd = fopen('D:\Zahra\111MyLesson\Az_neurosience\text.txt', 'rt');
mytext = '';
tl = fgets(fd);
lcount = 0;

while lcount < 5
    mytext = [mytext tl];
    tl = fgets(fd);
    lcount = lcount + 1;
end
fclose(fd);

% change score and key name in file
s3= 42+length(reply); mytext(s3:s3+length(key)-1)=key;
s1= 51+length(reply)+length(key); s2= 66+length(reply)+length(key);
mytext(s1)
mytext(s2)
mytext(s1)= int2str(reward);
temp= str2double(mytext(s2:end))+reward;
mytext(s2)= int2str(floor(temp/10)); mytext(s2+1)= int2str(mod(temp,10));
mytext(s2+1)

% show info on the screen
Screen('TextSize',wPtr, 30);
DrawFormattedText(wPtr, mytext, CenterX-200, CenterY-500, 255, 100);

% save in file
fd = fopen('D:\Zahra\111MyLesson\Az_neurosience\text.txt', 'wt');
fprintf(fd,'%s',mytext);
fclose(fd);
end