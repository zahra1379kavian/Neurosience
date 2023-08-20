function info_show_func(reply,wPtr,CenterX, CenterY)
% write info in the file
mytext= ['ID Number & Session Number: ',reply,newline,'Key Pressed: ','     ',newline,'Score: 0',newline,'Total Score: 00'];
fd = fopen('D:\Zahra\111MyLesson\Az_neurosience\text.txt', 'wt');
fprintf(fd,'%s',mytext);
fclose(fd);

% read file
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

% show on the screen
Screen('TextSize',wPtr, 30);
%DrawFormattedText(wPtr, mytext, 20, 30, 255, 100);
DrawFormattedText(wPtr, mytext, CenterX-200, CenterY-500, 255, 100);
end