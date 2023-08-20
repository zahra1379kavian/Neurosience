% Full Value Model
clc; clear; close all

% initial parameteres
x_size= 15; y_size= 15; n_iteration= 1500; n_trial= 3000;
learning_rate= 1; discount_factor= 1;
state_value= zeros(x_size,y_size);
reward= zeros(x_size,y_size);
m= cell(x_size,y_size); for i=1:x_size; for j= 1:y_size; m{i,j}= zeros(1,4); end; end
policy= cell(x_size,y_size); for i=1:x_size; for j= 1:y_size; policy{i,j}= zeros(1,4); end; end
x_mouse= zeros(n_trial,n_iteration);
y_mouse= zeros(n_trial,n_iteration);
trial_iter= zeros(1,n_trial); s=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% randomly choose mouse, cat, and platform position
platform_cordinate= [5 10];
cat_cordinate= [9 8];
cheese_cordinate= [13 7];
x_mouse_range= [1:4,6:10,11:12,14:15];
y_mouse_range= [1:6,9,11:15];
x_mouse(1,1)= x_mouse_range(randi([1 x_size-3]));
y_mouse(1,1)= y_mouse_range(randi([1 y_size-3]));
% design the game board
% f1= figure('Position', [10 50 1500 700]);
% d_mouse1= imread('D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW06_new\mouse1.jfif'); % scary mouse :)
% d_mouse3= imread('D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW06_new\mouse.jfif'); % happy mouse :)
% d_mouse2= imread('D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW06_new\mouse3.jfif'); % angry mouse :(
% GameBoard(x_mouse(1),y_mouse(1),platform_cordinate,cat_cordinate,d_mouse1,cheese_cordinate);
% subplot(6,6,[5,6,11,12,17,18]);
figure('Position', [10 50 1500 700]);
subplot(1,2,1)
imagesc(state_value);
colormap jet; colorbar;
set(gca,'XTick',[],'YTick',[]); %caxis([-1 1]);
subplot(6,6,[23,24,29,30,35,36]);
subplot(1,2,2)
[px,py]= gradient(flip(state_value));
quiver(1:15,1:15,px,py,'color',[0 1 0]);
set(gca,'XTick',[],'YTick',[]);
drawnow
hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial states probabilities
action_probability= initial_probability(x_size,y_size);
policy= action_probability;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for j= 1:n_trial
    clc
    j
    % randomly put the mouse on the board game
    if j~= 1
        x_mouse(j,1)= x_mouse_range(randi([1 x_size-3]));
        y_mouse(j,1)= y_mouse_range(randi([1 y_size-3]));
    end
    
    % update the board game
    %GameBoard(x_mouse(j,1),y_mouse(j,1),platform_cordinate,cat_cordinate,d_mouse1);hold on;
    
    for i= 1:n_iteration-1
        if x_mouse(j,i) ~= cat_cordinate(1) || y_mouse(j,i) ~= cat_cordinate(2)
            if x_mouse(j,i) ~= platform_cordinate(1) || y_mouse(j,i) ~= platform_cordinate(2)
                if x_mouse(j,i) ~= cheese_cordinate(1) || y_mouse(j,i) ~= cheese_cordinate(2)
                    
                    % update the policy, m (using gradient descent), action(I),
                    % and delta (use in update state value)
                    
                    [X, I, direction, avg_value]= X_calcu_full_model(x_mouse(j,i),y_mouse(j,i),action_probability,...
                        state_value,policy);
                    
                    delta= reward(16-y_mouse(j,i),x_mouse(j,i))+ discount_factor*X- state_value(16-y_mouse(j,i),x_mouse(j,i));
                    
                    state_value(16-y_mouse(j,i),x_mouse(j,i))= state_value(16-y_mouse(j,i),x_mouse(j,i))+...
                        learning_rate* delta;
                    
                    [policy, m]= policy_update(x_mouse(j,i),y_mouse(j,i),I,policy,m,learning_rate,state_value,avg_value,direction);
                    
                    
                    % change the mouse position
                    if I==1 % above
                        y_mouse(j,i+1)= min([y_mouse(j,i)+1 y_size]);
                        x_mouse(j,i+1)= x_mouse(j,i);
                    elseif I==2 % below
                        y_mouse(j,i+1)= max([y_mouse(j,i)-1 1]);
                        x_mouse(j,i+1)= x_mouse(j,i);
                    elseif I==3 % left
                        x_mouse(j,i+1)= max([x_mouse(j,i)-1 1]);
                        y_mouse(j,i+1)= y_mouse(j,i);
                    elseif I==4 % right
                        x_mouse(j,i+1)= min([x_mouse(j,i)+1 x_size]);
                        y_mouse(j,i+1)= y_mouse(j,i);
                    end
                    % If your system support high graphic program, you can
                    % uncomment next line.
                    %GameBoard(x_mouse(j,i+1),y_mouse(j,i+1),platform_cordinate,cat_cordinate,d_mouse1,cheese_cordinate);
                    
                    hold on
                    %if mod(i,50)== 0
                    %plot the mouse pathway
                    %plot(x_mouse(j,1:i+1),y_mouse(j,1:i+1),'r','linewidth',1);
                    %drawnow
                    %                 thisFrame = getframe(gcf);
                    %                 myMovie(s) = thisFrame;
                    %                 s=s+1;
                    %end
                    %title("Trial Number: "+j+" ,Iteration Number: "+i+" ,Learning Rate: "+learning_rate+...
                    %   " ,Discount Factor: "+discount_factor,'interpreter','latex')
                else
                    % if reach the platform-cheese, reward and the state value change
                    state_value(16-y_mouse(j,i),x_mouse(j,i))= 3;
                    reward(16-y_mouse(j,i),x_mouse(j,i))= 3;
                    %d_mouse= imread('D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW06\mouse.jfif'); % happy mouse :)
                    % redesign game board
                    %GameBoard(x_mouse(j,i),y_mouse(j,i),platform_cordinate,cat_cordinate,d_mouse3,cheese_cordinate);
                    %plot(x_mouse(j,1:i),y_mouse(j,1:i),'r','linewidth',1);
                    %drawnow
                    %hold on
                    trial_iter(j)= i;
                    % get this frame
                    %                     thisFrame = getframe(gcf);
                    %                     myMovie(s) = thisFrame;
                    %                     s=s+1;
                    break;
                end
            else
                % if reach the platform, reward and the state value change
                state_value(16-y_mouse(j,i),x_mouse(j,i))= 1;
                reward(16-y_mouse(j,i),x_mouse(j,i))= 1;
                % redesign game board
                %GameBoard(x_mouse(j,i),y_mouse(j,i),platform_cordinate,cat_cordinate,d_mouse3,cheese_cordinate);
                %plot(x_mouse(j,1:i),y_mouse(j,1:i),'r','linewidth',1);
                %drawnow
                trial_iter(j)= i;
                %                 if mod(i,20)==0
                %                     % get this frame
                %                     thisFrame = getframe(gcf);
                %                     myMovie(s) = thisFrame;
                %                     s=s+1;
                %                 end
                break;
            end
        else
            % if reach the cat, reward and the state value change
            reward(16-y_mouse(j,i),x_mouse(j,i))= -1;
            state_value(16-y_mouse(j,i),x_mouse(j,i))= -1;
            
            % redesign game board
            %GameBoard(x_mouse(j,i),y_mouse(j,i),platform_cordinate,cat_cordinate,d_mouse2,cheese_cordinate);
            %plot(x_mouse(j,1:i),y_mouse(j,1:i),'r','linewidth',1);
            %drawnow
            %             if mod(i,20)==0
            %                 % get this frame
            %                 thisFrame = getframe(gcf);
            %                 myMovie(s) = thisFrame;
            %                 s=s+1;
            %             end
            break;
        end
        
        %         if mod(i,20)==0
        %             thisFrame = getframe(gcf);
        %             myMovie(s) = thisFrame;
        %             s=s+1;
        %         end
    end
    % gradient and contour plot of the learned values
    %subplot(6,6,[5,6,11,12,17,18]);
    if mod(j,20)==0
        subplot(1,2,1)
        imagesc(imgaussfilt(state_value));
        colormap jet;
        colorbar;
        set(gca,'XTick',[],'YTick',[]); %caxis([-1 1]);
        %subplot(6,6,[23,24,29,30,35,36]);
        subplot(1,2,2)
        [px,py]= gradient(flip(state_value));
        quiver(1:15,1:15,px,py,'color',[0 1 0]);
        sgtitle("Trial Number: "+j+" ,Iteration Number: "+i+" ,Learning Rate: "+learning_rate+...
            " ,Discount Factor: "+discount_factor,'interpreter','latex')
        drawnow
        thisFrame = getframe(gcf);
        myMovie(s) = thisFrame;
        s=s+1;
        hold on
    end
end
%%
% Write the videos
writerObj = VideoWriter("demo2");
writerObj.FrameRate = 10;
open(writerObj);
numberOfFrames = length(myMovie);
for frameNumber = 1 : numberOfFrames
    writeVideo(writerObj, myMovie(frameNumber));
end
close(writerObj);
%%
% the number of iteration used in each trial
figure
stairs(trial_iter(1:j))
ylabel('number of iteration','interpreter','latex')
xlabel('trial number','interpreter','latex')
xlim([1 j])
%%
clc
figure
s= 1;
for i= 1:10:n_trial
    x= x_mouse(i,:); y= y_mouse(i,:);
    tmp_x= x(x~=0); tmp_y= y(y~=0);
    scatter(tmp_x(length(tmp_x)),tmp_y(length(tmp_y)),'MarkerFaceColor','b');
    hold on
    scatter(tmp_x(1),tmp_y(1),'fill','MarkerFaceColor','r');
    hold on
    %line([tmp_x(1) tmp_x(length(tmp_x))],[tmp_y(1) tmp_y(length(tmp_y))],'Color','g','LineStyle','--')
    plot(tmp_x,tmp_y,'--g')
    xlim([0 15]); ylim([0 15]);
    title("Pathway trial number: "+num2str(i),'interpreter','latex')
    drawnow
    pause(1)
    hold off
    thisFrame = getframe(gcf);
    myMovie(s) = thisFrame;
    s=s+1;
end

%%
% Write the videos
writerObj = VideoWriter("Pathway_demo2");
writerObj.FrameRate = 10;
open(writerObj);
numberOfFrames = length(myMovie);
for frameNumber = 1 : numberOfFrames
    writeVideo(writerObj, myMovie(frameNumber));
end
close(writerObj);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
clc; clear; close all

% initial parameteres
x_size= 15; y_size= 15; n_iteration= 1500; n_trial= 2000;
learning_rate= [0.1:0.2:1,1]; discount_factor= [0.1:0.2:1,1];
avg_step= zeros(length(learning_rate),length(discount_factor));
% randomly choose mouse, cat, and platform position
platform_cordinate= [5 11];
cat_cordinate= [9 8];
cheese_cordinate= [13 7];
x_mouse_range= [1:4,6:10,11:12,14:15];
y_mouse_range= [1:6,9,11:15];
figure('Position', [10 50 1500 700]);

for lr= 1:length(learning_rate)
    for ds= 1:length(discount_factor)
        % initial parameteres
        hold off
        state_value= zeros(x_size,y_size);
        reward= zeros(x_size,y_size);
        m= cell(x_size,y_size); for i=1:x_size; for j= 1:y_size; m{i,j}= zeros(1,4); end; end
        policy= cell(x_size,y_size); for i=1:x_size; for j= 1:y_size; policy{i,j}= zeros(1,4); end; end
        x_mouse= zeros(n_trial,n_iteration);
        y_mouse= zeros(n_trial,n_iteration);
        trial_iter= zeros(1,n_trial); s=1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % randomly choose mouse, cat, and platform position
        x_mouse(1,1)= x_mouse_range(randi([1 x_size-3]));
        y_mouse(1,1)= y_mouse_range(randi([1 y_size-3]));
        
        subplot(1,2,1)
        imagesc(state_value);
        colormap jet; colorbar;
        set(gca,'XTick',[],'YTick',[]); %caxis([-1 1]);
        subplot(6,6,[23,24,29,30,35,36]);
        subplot(1,2,2)
        [px,py]= gradient(flip(state_value));
        quiver(1:15,1:15,px,py,'color',[0 1 0]);
        set(gca,'XTick',[],'YTick',[]);
        drawnow
        hold on
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % initial states probabilities
        action_probability= initial_probability(x_size,y_size);
        policy= action_probability;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for j= 1:n_trial
            clc
            j
            % randomly put the mouse on the board game
            if j~= 1
                x_mouse(j,1)= x_mouse_range(randi([1 x_size-3]));
                y_mouse(j,1)= y_mouse_range(randi([1 y_size-3]));
            end
            
            % update the board game
            %GameBoard(x_mouse(j,1),y_mouse(j,1),platform_cordinate,cat_cordinate,d_mouse1);hold on;
            
            for i= 1:n_iteration-1
                if x_mouse(j,i) ~= cat_cordinate(1) || y_mouse(j,i) ~= cat_cordinate(2)
                    if x_mouse(j,i) ~= platform_cordinate(1) || y_mouse(j,i) ~= platform_cordinate(2)
                        if x_mouse(j,i) ~= cheese_cordinate(1) || y_mouse(j,i) ~= cheese_cordinate(2)
                            
                            % update the policy, m (using gradient descent), action(I),
                            % and delta (use in update state value)
                            
                            [X, I, direction, avg_value]= X_calcu_full_model(x_mouse(j,i),y_mouse(j,i),action_probability,...
                                state_value,policy);
                            
                            delta= reward(16-y_mouse(j,i),x_mouse(j,i))+ discount_factor(ds)*X- state_value(16-y_mouse(j,i),x_mouse(j,i));
                            
                            state_value(16-y_mouse(j,i),x_mouse(j,i))= state_value(16-y_mouse(j,i),x_mouse(j,i))+...
                                learning_rate(lr)* delta;
                            
                            [policy, m]= policy_update(x_mouse(j,i),y_mouse(j,i),I,policy,m,learning_rate(lr),state_value,avg_value,direction);
                            
                            
                            % change the mouse position
                            if I==1 % above
                                y_mouse(j,i+1)= min([y_mouse(j,i)+1 y_size]);
                                x_mouse(j,i+1)= x_mouse(j,i);
                            elseif I==2 % below
                                y_mouse(j,i+1)= max([y_mouse(j,i)-1 1]);
                                x_mouse(j,i+1)= x_mouse(j,i);
                            elseif I==3 % left
                                x_mouse(j,i+1)= max([x_mouse(j,i)-1 1]);
                                y_mouse(j,i+1)= y_mouse(j,i);
                            elseif I==4 % right
                                x_mouse(j,i+1)= min([x_mouse(j,i)+1 x_size]);
                                y_mouse(j,i+1)= y_mouse(j,i);
                            end
                            % If your system support high graphic program, you can
                            % uncomment next line.
                            %GameBoard(x_mouse(j,i+1),y_mouse(j,i+1),platform_cordinate,cat_cordinate,d_mouse1,cheese_cordinate);
  
                        else
                            % if reach the platform-cheese, reward and the state value change
                            state_value(16-y_mouse(j,i),x_mouse(j,i))= 3;
                            reward(16-y_mouse(j,i),x_mouse(j,i))= 3;
                            break;
                        end
                    else
                        % if reach the platform, reward and the state value change
                        state_value(16-y_mouse(j,i),x_mouse(j,i))= 1;
                        reward(16-y_mouse(j,i),x_mouse(j,i))= 1;
                        break;
                    end
                else
                    % if reach the cat, reward and the state value change
                    reward(16-y_mouse(j,i),x_mouse(j,i))= -1;
                    state_value(16-y_mouse(j,i),x_mouse(j,i))= -1;

                    break;
                end

            end
            % gradient and contour plot of the learned values
            if mod(j,20)==0
                subplot(1,2,1)
                imagesc(imgaussfilt(state_value));
                colormap jet;
                colorbar;
                set(gca,'XTick',[],'YTick',[]);
                subplot(1,2,2)
                [px,py]= gradient(flip(state_value));
                quiver(1:15,1:15,px,py,'color',[0 1 0]);
                sgtitle("Trial Number: "+j+" ,Iteration Number: "+i+" ,Learning Rate: "+learning_rate(lr)+...
                    " ,Discount Factor: "+discount_factor(ds),'interpreter','latex')
                drawnow
%                 thisFrame = getframe(gcf);
%                 myMovie(s) = thisFrame;
%                 s=s+1;
                %hold on
            end
        end
        for i= 1:100
            avg_step(lr,ds)= avg_step(lr,ds) + nnz(x_mouse(n_trial+1-i,:))/100;
        end
        
    end
end
%%

figure
imagesc(learning_rate,discount_factor,log(avg_step))
colormap summer
title('Effect of Learning Rate and Discount Factor in the Learning Procedure','interpreter','latex')
xlabel('Learning Rate','interpreter','latex')
ylabel('Discount Factor','interpreter','latex')
colorbar
axis equal
xlim([0 1]); ylim([0 1]);
set(gca,'YDir','normal')

