datcom=readtable('datatable.csv')
cn=readtable('cn.csv')

ax1=subplot(5,1,1)
plot(ax1,cn.Var1,cn.cn_b,'LineWidth',2)
hold on
plot(ax1,datcom.alpha,datcom.cn_b,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax1,'C_{n_{\beta}}')
hold off
leg=legend('estimated value','actual value','Orientation','Horizontal','Position','NorthEast')
pos=get(ax1,'Position')
pos2=get(leg,'Position')
pos2(2)=pos2(2)+0.08
set(leg,'Position',pos2)

ax2=subplot(5,1,2)
plot(ax2,cn.Var1,cn.cn_p,'LineWidth',2)
hold on\subsection{Longitudinal and lateral data}
plot(ax2,datcom.alpha,datcom.cn_p,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax2,'C_{n_{p}}')
hold off
pos(2)=pos(2)-0.18
set(ax2,'Position',pos)

ax3=subplot(5,1,3)
plot(ax3,cn.Var1,cn.cn_r,'LineWidth',2)
hold on
plot(ax3,datcom.alpha,datcom.cn_r,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax3,'C_{n_{r}}')
hold off
pos(2)=pos(2)-0.18
set(ax3,'Position',pos)

ax4=subplot(5,1,4)
plot(ax4,cn.Var1,cn.cn_da,'LineWidth',2)
hold on
plot(ax4,datcom.alpha,datcom.cn_da,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax4,'C_{n_{\delta_{a}}}')
hold off
pos(2)=pos(2)-0.18
set(ax4,'Position',pos)

ax5=subplot(5,1,5)
plot(ax5,cn.Var1,cn.cn_dr,'LineWidth',2)
hold on
plot(ax5,datcom.alpha,datcom.cn_dr,'LineWidth',2,'Color','r','LineStyle','--')
ylabel(ax5,'C_{n_{\delta_{r}}}')
hold off
pos(2)=pos(2)-0.18
set(ax5,'Position',pos)

linkaxes([ax1,ax2,ax3,ax4,ax5],'x')
set(ax1,'xticklabel',{[]})
set(ax2,'xticklabel',{[]})
set(ax3,'xticklabel',{[]})
set(ax4,'xticklabel',{[]})
axis(ax1,'tight')
axis(ax2,'tight')
axis(ax3,'tight')
axis(ax4,'tight')
axis(ax5,'tight')
xlabel('\alpha','FontSize',15)