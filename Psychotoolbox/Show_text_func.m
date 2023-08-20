function Show_text_func(wPtr,txt,opt)
Screen('TextFont',wPtr,opt.font);
Screen('TextSize',wPtr,opt.size);
Screen('DrawText',wPtr,txt,opt.location(1),opt.location(2),opt.color);
Screen('Flip',wPtr)
end