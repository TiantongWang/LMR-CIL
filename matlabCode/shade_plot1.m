function shade_plot2(x,y,y_sem,color)
y_u=y+y_sem;
y_d=y-y_sem;
x_shade=[x x(end:-1:1)];
y_shade=[y_u y_d(end:-1:1)];
% fill(x_shade,y_shade,color,'FaceAlpha',0.1,'LineStyle','none');
fill(x_shade,y_shade,color,'LineStyle','none');
% fill(x_shade,y_shade,color,'FaceAlpha',0.5);
% fill(x_shade,y_shade,'w');
hold on;
plot(x,y,'r-','linewidth',1); % plotyy
