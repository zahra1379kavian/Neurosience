function ITI= set_ITI_func(Null, good_choice, bad_choice, wrong_choice, trial_reject, TP_reject, reward_time)
if Null
    ITI= 1.5; return;
elseif good_choice
    ITI= 1.5; return;
elseif bad_choice
    ITI= 1.5; return;
elseif wrong_choice
    ITI= 1.5; return;
elseif trial_reject
    if reward_time
        ITI= 1.5; return;
    else
        ITI= 0.2; return;
    end
elseif TP_reject
    ITI= 0.2; return;
% else
%     ITI= 1.5;
end

end