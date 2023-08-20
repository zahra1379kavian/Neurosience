function GameBoard(X_position_mouse,Y_position_mouse,platform_cordinate,cat_cordinate,d_mouse)
subplot(6,6,[1:4,7:10,13:16,19:22,25:28,31:34])
set(gca,'XTick',[],'YTick',[]);
back_color= ones(16,16);
pcolor(1:16,1:16,back_color);
grid on
caxis([0 1])
colormap gray
set(gca,'XTick',[],'YTick',[])
hold on
d= imread('D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW06_new\platform.jfif');
image(flipud(d), 'XData', [platform_cordinate(1) platform_cordinate(1)+1],...
    'YData', [platform_cordinate(2) platform_cordinate(2)+1]);
hold on
d= imread('D:\Zahra\111MyLesson\Ad_Neuro\Homework\HW06_new\cat.png');
image(flipud(d), 'XData', [cat_cordinate(1) cat_cordinate(1)+1]...
    , 'YData', [cat_cordinate(2) cat_cordinate(2)+1]);
hold on
image(flipud(d_mouse), 'XData', [X_position_mouse X_position_mouse+1], 'YData', [Y_position_mouse Y_position_mouse+1]);
end