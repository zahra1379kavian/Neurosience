function Draw_cross_func(wPtr,opt)
len= opt.length;
lines= [-len len 0 0;0 0 -len len];
Screen('DrawLines',wPtr,lines,opt.width,opt.color,[opt.location(1) opt.location(2)],opt.smooth);
end