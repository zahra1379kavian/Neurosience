function reward= set_reward_func(null, good_choice, bad_choice, wrong_choice,trial_reject, TP_reject, reward_time)
if null
    reward= 0; return;
elseif good_choice
    reward= 3; return;
elseif bad_choice
    reward= 1; return;
elseif wrong_choice
    reward= 1; return;
elseif trial_reject
    if reward_time
        reward= 2; return;
    else
        reward= 0; return;
    end
elseif TP_reject
    reward= 0; return;
end

end