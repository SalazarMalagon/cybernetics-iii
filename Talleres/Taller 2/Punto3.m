clear,clc, close all
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])
% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=0.3; sig=10; r=28; a=0.35; w=1;

% Corrected: This is a 2D system, not 3D
Lorenz = @(t,x)([ x(2)      ; ...
                  x(1)-(x(1)^3)-(a*x(2))+(b*cos(w*t))]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
for j=1:100  % training trajectories
    % Corrected: Generate 2D initial conditions instead of 3D
    x0=3*(rand(2,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    % Corrected: Plot 2D trajectory instead of 3D
    plot(y(:,1),y(:,2)), hold on
    plot(x0(1),x0(2),'ro')
end
grid on

%%
net = feedforwardnet([10 10 10 10]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%%
figure(2)
% Corrected: Generate 2D initial conditions
x0=1*(rand(2,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
% Corrected: Plot 2D trajectory
plot(y(:,1),y(:,2)), hold on
plot(x0(1),x0(2),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
% Corrected: Plot 2D trajectory
plot(ynn(:,1),ynn(:,2),':','Linewidth',[2])

figure(3)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
% Removed subplot(3,2,5) since we only have 2 variables

figure(2)
% Corrected: Generate 2D initial conditions
x0=1*(rand(2,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
% Corrected: Plot 2D trajectory
plot(y(:,1),y(:,2)), hold on
plot(x0(1),x0(2),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
% Corrected: Plot 2D trajectory
plot(ynn(:,1),ynn(:,2),':','Linewidth',[2])

figure(3)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
% Removed subplot(3,2,6) since we only have 2 variables

%%
figure(3)
subplot(3,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])
legend('Duffing Attractor','NN')