function [null, good_choice, bad_choice, wrong_choice, trial_reject, TP_reject]= set_condition_func2(state, keyCode, x_mouse,y_mouse, rects, s3)
% Null: no press key
% target absent
good_choice=0; bad_choice=0; wrong_choice=0; trial_reject=0; null=0; TP_reject=0; key=KbName(keyCode);
if isempty(key)
    null= 1;
elseif ~state & ~strcmp(key, 'space') %X
    bad_choice=1; return;
elseif ~state & strcmp(key, 'space') %space
    trial_reject=1; return;
elseif state & strcmp(key, 'space') %space
    TP_reject=1; return;
elseif state & ~strcmp(key, 'space') %X
    for i=1: size(rects, 2)
        if  rects(1,s3)<= x_mouse & rects(3,s3)>= x_mouse & rects(2,s3)<= y_mouse & rects(4,s3)>= y_mouse
            good_choice=1; return;
        else
            wrong_choice=1; return;
        end
    end
    
end