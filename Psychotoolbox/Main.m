clc
clear
close all

% Load Images
cd= 'D:\Zahra\111MyLesson\Az_neurosience\Lab2\Assignment2\Assignment2_fractals';
image_num= 48;

for i=1:image_num
    matFilename= sprintf('%d.jpeg',i);
    mat(:,:,:,i)= imread(fullfile(cd, matFilename));
end

% Open Pshychtoolbox
%PsychDebugWindowConfiguration
Screen('Preference', 'SkipSyncTests', 1);
[wPtr,rect]= Screen('OpenWindow',max(Screen('screens')),[0 0 0]);
CenterX= rect(3)/2; CenterY= rect(4)/2;

% Get user information
reply=Ask(wPtr,'Enter your ID and session number.',[255 255 255],[0 0 0],'GetChar',[460 240 1260 840],'center',50);
WaitSecs(1);
Screen('Flip',wPtr);

% Start
txt= 'Press Enter to Begin.'; opt.font= 'Helvetica'; opt.size= 100; opt.location= [CenterX-480 CenterY]; opt.color= [255 255 255];
Show_text_func(wPtr,txt,opt);

% Wait to press Enter
[secs, keyCode, deltaSecs] = KbWait();
% Check use correct key.
a = KbName(keyCode);
if strcmp(KbName(keyCode),'return')
    Screen('Flip',wPtr);
else
    while strcmp(KbName(keyCode),'return')==0
        txt= 'Wrong Key!';
        Show_text_func(wPtr,txt,opt);
        WaitSecs(1);
        Screen('Flip',wPtr);
        txt= 'Press Enter to Begin.';
        Show_text_func(wPtr,txt,opt);
        WaitSecs(1);
        [secs, keyCode] = KbWait();
    end
end

% Show fixation point
opt.width=5; opt.color=[0 0 255]; opt.location=[CenterX CenterY]; opt.smooth=0; opt.length=100;
Draw_cross_func(wPtr,opt);
WaitSecs(randi([300 500])*0.001);
% show info on right top screen
info_show_func(reply,wPtr,CenterX, CenterY);
WaitSecs(0.005);
Screen('Flip',wPtr);

opt.color= [255 0 0];

% Wait
WaitSecs(1);

% Show images
trl_num= 24; t2wait= 3; trl_img_num= [3 5 7 9]; reward_time=0; s4= randi([2,4],1,1); %For reward time in TA and rejection

%%% Detemine which trials have a good image or not
% state= ones(1,trl_num); s= randperm(trl_num,trl_num/2); state(s)= 0;

trial_reject=0;
%%% Main loop
ShowCursor('Hand');

keyIsDown=0; buttons=[]; s2=0;

for i= 1:trl_num
    % randomly choose the zero angle of images
    zero_angle= randi([0, 181],1,1);
    
    % Set our parameters
    [DS_trials, state, TA_bad_fractal, TP_bad_fractal, TP_good_fractal, good_label, bad_label, TA_trials, TP_trials]= Randomization(trl_num,trl_img_num);
    if state(i)==1
        s1= sum(DS_trials(state(1:i-1)==1)-1);
        num= zeros(1,DS_trials(i));
        s2=randi(DS_trials(i),1,1);
        s3= length(find(state(1:i-1)==1));
        num(s2)= TP_good_fractal(s3+1);
        num(num==0)= TP_bad_fractal(s1+1:s1+DS_trials(i)-1);
    else
        s1= sum(DS_trials(state(1:i-1)==0));
        num= TA_bad_fractal(s1+1:s1+DS_trials(i));
    end
    
    [rects, imageWidth, imageHight, n, deg]= show_image_func(wPtr,mat,DS_trials(i),num,CenterX,CenterY,good_label,zero_angle); %%%%%%%%%%%
    tStart= Screen('Flip',wPtr);
    
    timedout= false;
    
    while ~timedout
        % track mouse position
        theX= CenterX;
        theY= CenterY;
        SetMouse(theX,theY,wPtr);
        while (1)
            [x,y,buttons] = GetMouse(wPtr);
            if buttons(1)
                break;
            end
        end
        % Loop and track the mouse, drawing the contour
        [theX,theY] = GetMouse(wPtr);
        thePoints = [theX theY];
        Screen(wPtr,'DrawLine',255,theX,theY,theX,theY); keyIsDown=0;
        
        while ~keyIsDown
            [x,y,buttons] = GetMouse(wPtr);
            [keyIsDown, keyTime, keyCode]= KbCheck;
            if (x ~= theX || y ~= theY)
                rects= show_image_func(wPtr,mat,DS_trials(i),num,CenterX,CenterY,good_label,zero_angle);
                Time= Screen('Flip',wPtr);
                thePoints = [thePoints ; x y]; %#ok<AGROW>
                [numPoints, two]=size(thePoints);
                Screen(wPtr,'DrawLine',128,thePoints(numPoints-1,1),thePoints(numPoints-1,2),thePoints(numPoints,1),thePoints(numPoints,2),10);
                theX = x; theY = y;
                if((Time - tStart) > t2wait), timedout = true; break; end
            end
            
        end
        x_mouse(i)=thePoints(end,1); y_mouse(i)=thePoints(end,2);
        
        [null(i), good_choice(i), bad_choice(i), wrong_choice(i), trial_reject(i), TP_reject(i)]= set_condition_func2(state(i), keyCode, x_mouse(i), y_mouse(i), rects, s2);
        while KbCheck; end
        
        %break;
        if((Time - tStart) > t2wait), timedout = true; else, break; end
        
    end % while ~timeout
    
    % save response time and the kyes are pressed
    if timedout==false
        rsp.RT{i} = keyTime - tStart;
        rsp.keyName{i} = KbName(keyCode);
        rsp.mouse_position{i}= thePoints;
    else
        rsp.RT{i} = 0;
        null(i)= 1; good_choice(i)=0; bad_choice(i)=0; wrong_choice(i)=0; trial_reject(i)=0; TP_reject(i)=0;
        % Show error
        txt= 'Too slow!'; opt.location= [CenterX-200  CenterY]; opt.color=[255 255 255]; opt.size= 100;
        % Beep sound
        beep = MakeBeep(400,.5);
        Snd('Open');
        Show_text_func(wPtr,txt,opt);
        Snd('Play',beep);
        WaitSecs(2);
    end
    
    % set the reward time for TA and rejection
    if length(find(trial_reject))==s4
        reward_time=1;
    end
    
    % set inter trial interval for this trial
    ITI= set_ITI_func(null(i), good_choice(i), bad_choice(i), wrong_choice(i), trial_reject(i), TP_reject(i), reward_time);
    reward= set_reward_func(null(i), good_choice(i), bad_choice(i), wrong_choice(i), trial_reject(i), TP_reject(i), reward_time);
    key= KbName(keyCode);
    if strcmp(key, 'x')
        key= 'x_key';
    elseif isempty(key)
        key='     ';
    end
    
    reward_show_func(wPtr,reward,reply,CenterX,CenterY,key);
    WaitSecs(0.5)
    Screen('Flip',wPtr);
    % Show the reward :)
    if reward_time
        img= imread('Monkey1.jpg');
        T= Screen('MakeTexture',wPtr,img);
        Screen('DrawTextures',wPtr,T,[]);
        WaitSecs(1)
        Screen('Flip',wPtr);
    end
    WaitSecs(ITI)
    Screen('Flip',wPtr);
    
    %show cross
    opt.location= [CenterX CenterY]; opt.color=[0 0 255];
    Draw_cross_func(wPtr,opt);
    Screen('Flip',wPtr);
    WaitSecs(randi([300 500])*0.001);
    
    keyIsDown=0; buttons=[]; j=1;
    
    if reward_time==1, trial_reject= zeros(1,length(trial_reject)); reward_time=0; end
end
sca

%%%% Results %%%%%
% Reaction Time Histogram
figure
hist(cell2mat(rsp.RT));
h = findobj(gca,'Type','patch'); h.FaceColor = [0 0.5 0.5]; h.EdgeColor = 'w';
xlabel('time','Interpreter','latex')
title('Reaction Time','Interpreter','latex')
% Confusion Matrix
figure
plotconfusion(state,good_choice+bad_choice)
%fontsize
set(findobj(gca,'type','text'),'fontsize',20)


subject= [];
subject.ID_Session= reply;
subject.reaction_time= rsp.RT;
subject.button_pressed= rsp.keyName;
subject.mouse_position= rsp.mouse_position;
subject.fractal_size= [imageWidth/n imageHight/n];
subject.fractals_position_on_display= deg;
subject.screen_size= [rect(3) rect(4)];
subject.DS= DS_trials;
subject.TA_trial_num= TA_trials;
subject.TP_trial_num= TP_trials;
subject.fractals_name.TP_bad_fractal= TP_bad_fractal;
subject.fractals_name.TP_good_fractal= TP_good_fractal;
subject.fractals_name.TA_bad_fractal= TA_bad_fractal;

save subject subject
