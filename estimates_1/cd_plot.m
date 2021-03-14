datcom=readtable('datatable.csv')
cd=readtable('cd.csv')

ax1=subplot(3,1,1)
plot(ax1,cd.Var1,cd.cd0,'LineWidth',2)
hold on
plot(ax1,datcom.alpha,datcom.cd0,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax1,'C_{D_{0}}')
hold off
leg=legend('estimated value','actual value','Orientation','Horizontal','Position','NorthEast')
pos=get(ax1,'Position')
pos2=get(leg,'Position')
pos2(2)=pos2(2)+0.09
set(leg,'Position',pos2)

ax2=subplot(3,1,2)
plot(ax2,cd.Var1,cd.cd_q,'LineWidth',2)
hold on
plot(ax2,datcom.alpha,datcom.cd_q,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax2,'C_{D_{q}}')
hold off
pos(2)=pos(2)-0.27
set(ax2,'Position',pos)

ax3=subplot(3,1,3)
plot(ax3,cd.Var1,cd.cd_de,'LineWidth',2)
hold on
plot(ax3,datcom.alpha,datcom.cd_de,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax3,'C_{D_{\delta_{e}}}')
hold off
pos(2)=pos(2)-0.27
set(ax3,'Position',pos)

linkaxes([ax1,ax2,ax3],'x')
set(ax1,'xticklabel',{[]})
set(ax2,'xticklabel',{[]})
axis(ax1,'tight')
axis(ax2,'tight')
axis(ax3,'tight')
xlabel('\alpha','FontSize',15)