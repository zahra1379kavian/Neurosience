close all
clc
clear

x_size= 15; y_size= 15; n_iteration= 300; n_trial= 20;
x_mouse= zeros(1,n_iteration); y_mouse= zeros(1,n_iteration);
learning_rate= 0.1; discount_factor=1;
state_value= zeros(x_size,y_size); reward= zeros(x_size,y_size);


% design the game board
platform_cordinate= [3 12]; cat_cordinate= [7 10]; x_mouse_range= [1:2,4:6,8:15]; y_mouse_range= [1:2,4,6:15];
x_mouse(1)= x_mouse_range(randi([1 x_size-2])); y_mouse(1)= y_mouse_range(randi([1 y_size-2]));
GameBoard(x_mouse(1),y_mouse(1),platform_cordinate,cat_cordinate); hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial states probabilities
action_probability= initial_probability(x_size,y_size);
reward(16-cat_cordinate(2),cat_cordinate(1))= -1; reward(16-platform_cordinate(2),platform_cordinate(1))= 1;
state_value(16-cat_cordinate(2),cat_cordinate(1))= -1; state_value(16-platform_cordinate(2),platform_cordinate(1))= 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Dynamic_Programming__Model Free
% clc
% for j= 1:n_trial
%     %plot(x_mouse(1:i+1),y_mouse(1:i+1),'k','linewidth',1.5); hold on;
%     x_mouse= zeros(1,n_iteration); y_mouse= zeros(1,n_iteration);
%     x_mouse(1)= x_mouse_range(randi([1 x_size-2])); y_mouse(1)= y_mouse_range(randi([1 y_size-2]));
%     GameBoard(x_mouse(1),y_mouse(1),platform_cordinate,cat_cordinate); hold on;
%     for i= 1:n_iteration-1
%         if x_mouse(i) ~= cat_cordinate(1) || y_mouse(i) ~= cat_cordinate(2)
%             if x_mouse(i) ~= platform_cordinate(1) || y_mouse(i) ~= platform_cordinate(2)
%                 %%[M, I] = max(states_probability{x_mouse(i),y_mouse(i)});
%                 %idx= find(action_probability{x_mouse(i),y_mouse(i)}==max(action_probability{x_mouse(i),y_mouse(i)}));
%                 %delta= reward(x_mouse(i),y_mouse(i))+discount_factor*state_value()-state_value();
%                 [delta, idx]= delta_calcu_model_free(x_mouse(i),y_mouse(i),discount_factor,state_value,reward);
%                 I= idx(randi([1 length(idx)]));
%                 delta(I)
%                 state_value(16-y_mouse(i),x_mouse(i))= state_value(16-y_mouse(i),x_mouse(i))+learning_rate*delta(I);
%                 
%                 if I==1 %above
%                     y_mouse(i+1)= min([y_mouse(i)+1 y_size-1]); x_mouse(i+1)= x_mouse(i);
%                 elseif I==2 % below
%                     y_mouse(i+1)= max([y_mouse(i)-1 1]); x_mouse(i+1)= x_mouse(i);
%                 elseif I==3 % left
%                     x_mouse(i+1)= max([x_mouse(i)-1 1]); y_mouse(i+1)= y_mouse(i);
%                 elseif I==4 % right
%                     x_mouse(i+1)= min([x_mouse(i)+1 x_size-1]); y_mouse(i+1)= y_mouse(i);
%                 end
%                 %GameBoard(x_mouse(i+1),y_mouse(i+1),platform_cordinate,cat_cordinate);
%                 hold on
%                 plot(x_mouse(1:i+1),y_mouse(1:i+1),'r','linewidth',1);
%                 drawnow
%                 hold on
%                 %GameBoard(x_mouse,y_mouse);
%             else
%                 
%                 %reward(x_mouse(i),y_mouse(i))= 1;
%                 GameBoard(x_mouse(i),y_mouse(i),platform_cordinate,cat_cordinate);
%                 plot(x_mouse(1:i),y_mouse(1:i),'r','linewidth',1);
%                 drawnow
%                 hold on
%                 break;
%             end
%         else
%             %reward(x_mouse(i),y_mouse(i))= -1;
%             GameBoard(x_mouse(i),y_mouse(i),platform_cordinate,cat_cordinate);
%             plot(x_mouse(1:i),y_mouse(1:i),'r','linewidth',1);
%             drawnow
%             hold on
%             break;
%         end
%         pause(0.01)
%     end
%     pause(0.5)
% end
%%
% Full Value Model
clc; clear; close all

x_size= 15; y_size= 15; n_iteration= 400; n_trial= 20;
%x_mouse= zeros(1,n_iteration); y_mouse= zeros(1,n_iteration);
learning_rate= 1; discount_factor=1;
state_value= zeros(x_size,y_size); reward= zeros(x_size,y_size);
m= cell(x_size,y_size); for i=1:x_size; for j= 1:y_size; m{i,j}= zeros(1,4); end; end
policy= cell(x_size,y_size); for i=1:x_size; for j= 1:y_size; policy{i,j}= zeros(1,4); end; end
x_mouse= zeros(n_trial,n_iteration); y_mouse= zeros(n_trial,n_iteration);

% design the game board
platform_cordinate= [3 12]; cat_cordinate= [7 10]; x_mouse_range= [1:2,4:6,8:14]; y_mouse_range= [1:9,11,13:14];
x_mouse(1,1)= x_mouse_range(randi([1 x_size-3])); y_mouse(1,1)= y_mouse_range(randi([1 y_size-3]));
figure('Position', [10 50 1500 700]);
d_mouse= imread('mouse1.jfif');
GameBoard(x_mouse(1),y_mouse(1),platform_cordinate,cat_cordinate,d_mouse); hold on;
subplot(6,6,[5,6,11,12,17,18]); pcolor(1:15,15:-1:1,imgaussfilt(state_value')); shading interp; 
colormap gray; colorbar;
subplot(6,6,[23,24,29,30,35,36]); [px,py]= gradient(flip(state_value)); quiver(1:15,1:15,px,py); hold on; set(gca,'XTick',[],'YTick',[])
scatter(platform_cordinate(1),platform_cordinate(2),'fill','MarkerFaceColor','k');
scatter(cat_cordinate(1),cat_cordinate(2),'fill','MarkerFaceColor','r')
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial states probabilities
action_probability= initial_probability(x_size,y_size);
reward(16-cat_cordinate(2),cat_cordinate(1))= -1; reward(16-platform_cordinate(2),platform_cordinate(1))= 1;
%state_value(16-cat_cordinate(2),cat_cordinate(1))= -1; state_value(16-platform_cordinate(2),platform_cordinate(1))= 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trial_iter= zeros(1,n_trial); s=1; 



for j= 1:n_trial
    clc
    j
    if j~= 1; x_mouse(j,1)= x_mouse_range(randi([1 x_size-3])); y_mouse(j,1)= y_mouse_range(randi([1 y_size-3])); end
    d_mouse= imread('mouse1.jfif'); 
    GameBoard(x_mouse(j,1),y_mouse(j,1),platform_cordinate,cat_cordinate,d_mouse);hold on;
    
    for i= 1:n_iteration-1
        if x_mouse(j,i) ~= cat_cordinate(1) || y_mouse(j,i) ~= cat_cordinate(2)
            if x_mouse(j,i) ~= platform_cordinate(1) || y_mouse(j,i) ~= platform_cordinate(2)
                
                [delta(i),I, policy, m]= delta_calcu_full_model(x_mouse(j,i),y_mouse(j,i),action_probability,...
                    state_value,m,policy,reward,learning_rate);
                state_value(16-y_mouse(j,i),x_mouse(j,i))= reward(16-y_mouse(j,i),x_mouse(j,i))+...
                    discount_factor*delta(i);
              
                if I==1 % above
                    y_mouse(j,i+1)= min([y_mouse(j,i)+1 y_size]); x_mouse(j,i+1)= x_mouse(j,i);
                elseif I==2 % below
                    y_mouse(j,i+1)= max([y_mouse(j,i)-1 1]); x_mouse(j,i+1)= x_mouse(j,i);
                elseif I==3 % left
                    x_mouse(j,i+1)= max([x_mouse(j,i)-1 1]); y_mouse(j,i+1)= y_mouse(j,i);
                elseif I==4 % right
                    x_mouse(j,i+1)= min([x_mouse(j,i)+1 x_size]); y_mouse(j,i+1)= y_mouse(j,i);
                end
                d_mouse= imread('mouse1.jfif');
                %GameBoard(x_mouse(j,i+1),y_mouse(j,i+1),platform_cordinate,cat_cordinate,d_mouse);
                hold on
                plot(x_mouse(j,1:i+1),y_mouse(j,1:i+1),'r','linewidth',1);
                drawnow 

                hold on
                title("Trial Number: "+j+" ,Iteration Number: "+i+" ,Learning Rate: "+learning_rate+...
                    " ,Discount Factor: "+discount_factor,'interpreter','latex')

                %GameBoard(x_mouse,y_mouse);
            else
               
                state_value(16-y_mouse(j,i),x_mouse(j,i))= 1;
                reward(16-y_mouse(j,i),x_mouse(j,i))= 1;
                d_mouse= imread('mouse.jfif');
                GameBoard(x_mouse(j,i),y_mouse(j,i),platform_cordinate,cat_cordinate,d_mouse);
                plot(x_mouse(j,1:i),y_mouse(j,1:i),'r','linewidth',1);
                drawnow
                hold on
                trial_iter(j)= i;
                thisFrame = getframe(gcf);
                myMovie(s) = thisFrame;
                s=s+1;
                break;
            end
        else
            reward(16-y_mouse(j,i),x_mouse(j,i))= -1;
            state_value(16-y_mouse(j,i),x_mouse(j,i))= -1;
            d_mouse= imread('mouse3.jfif');
            GameBoard(x_mouse(j,i),y_mouse(j,i),platform_cordinate,cat_cordinate,d_mouse);
            plot(x_mouse(j,1:i),y_mouse(j,1:i),'r','linewidth',1);
            drawnow
            hold on
            thisFrame = getframe(gcf);
            myMovie(s) = thisFrame;
            s=s+1;
            break;
        end
        
        pause(0.001)
        thisFrame = getframe(gcf);
        myMovie(s) = thisFrame;
        s=s+1;
    end
    subplot(6,6,[5,6,11,12,17,18]); pcolor(1:15,15:-1:1,imgaussfilt(state_value')); shading interp; colormap gray; colorbar;
    subplot(6,6,[23,24,29,30,35,36]); [px,py]= gradient(flip(state_value)); quiver(1:15,1:15,px,py); hold on; 
    scatter(platform_cordinate(1),platform_cordinate(2),'fill','MarkerFaceColor','k');
    scatter(cat_cordinate(1),cat_cordinate(2),'fill','MarkerFaceColor','r')
%     plot(x_mouse(j,1:i),y_mouse(j,1:i),'r','linewidth',1);
%     drawnow
%     hold on
%     back_color(y_mouse(j,i),x_mouse(j,i))= 50/255;
%     pcolor(1:16,1:16,back_color);
%     %GameBoard(x_mouse(j,i),y_mouse(j,i),platform_cordinate,cat_cordinate,d_mouse,back_color);
    pause(0.001)
end

%%
%Write the videos
writerObj = VideoWriter("Greedy Policy");
writerObj.FrameRate = 10;
open(writerObj);
numberOfFrames = length(myMovie);
for frameNumber = 1 : numberOfFrames
    writeVideo(writerObj, myMovie(frameNumber));
end
close(writerObj);

%%
figure
stairs(trial_iter)
ylabel('number of iteration','interpreter','latex')
xlabel('trial number','interpreter','latex')
title('Greedy Policy','interpreter','latex')
%%
% figure
% [px,py]= gradient(flip(state_value));
% quiver(1:16,1:16,px,py);
% hold on
% scatter(platform_cordinate(1),platform_cordinate(2),'fill','MarkerFaceColor','k')
% scatter(cat_cordinate(1),cat_cordinate(2),'fill','MarkerFaceColor','r')

