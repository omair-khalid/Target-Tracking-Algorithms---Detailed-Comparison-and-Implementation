% Lab 2 - Extended Kalman Filter
%Yu Liu, Wajhat Akhtar, Omair Khalid
% Date - 03/10/2017
clear all
close all

numP = 300; %number of iterations
%Initialize x0
%x0 = [10;1;10;0];
x0 = [-200;80;-200;20]

%Process noise Standard Deviation
sigma_q = 1.0;
%observation noise Standard Deviation 
sigma_r = 5.0; 

deltaT = 0.5; %time step

%Covariance of Prior
P = 1000 * eye(4,4);

%State Transition Matrix (Markov kernel)
f = [1, deltaT;0 1];
F = [f zeros(2,2);zeros(2,2) f]; 

%Covariance Process Noise
q = [deltaT^4/4 deltaT^3/2;deltaT^3/2 deltaT^2];
Q = sigma_q^2 * [q ,zeros(2,2);zeros(2,2) q]; %process noise matrix


%Transform - State to Measurement Space
%H = [1,0,0,0;0,0,1,0];

%Covariance of Observation noise
%R = sigma_r^2 * [1,0;0,1];
R = diag([50^2 0.005^2]);


X = [];
Z = [];

%Simulator 
for i = 1:numP
    
    x = mvnrnd(F*x0,Q)'; %predicted state
    X = [X,x];
    range = sqrt(x0(1)^2+x0(3)^2);
    bearing = atan2(x0(3),x0(1));
    y = [range;bearing];
    
    z = mvnrnd(y,R)'; %measurement 
    Z = [Z,z];
    x0 = x;
end


figure(1), hold on;
plot(X(1,:),X(3,:),'bx');
hold on
plot(Z(1,:).*cos(Z(2,:)),Z(1,:).*sin(Z(2,:)),'g+');
  
hold on
%%
%Kalman Filter 
pause(4);
for j = 1:numP
     %Kalman statr prediction
    range = sqrt(x0(1)^2+x0(3)^2);
    bearing = atan2(x0(3),x0(1));
    y0 = [range; bearing];
    J = [cos(bearing) 0 sin(bearing) 0;
     -sin(bearing)/range 0 cos(bearing)/range 0]; 
    m = F*x0;
    P = F*P*F' + Q;
    K = P*J' * inv(J*P*J' + R); 
    Updated_mean = m + K * (Z(:,j) - y0);
    Updated_P = (eye(4,4) - K*J)*P;
    x0 = Updated_mean;
    P = Updated_P;
    plot(x0(1),x0(3),'r+');
    hold on 
     
    pause(0.1);
    
  title( ['Extended Kalman Filter Tracking - Time Step = ' num2str( j )] )
legend('Simulator','Measurement', 'Kalman Update','Location','southeast')

end

    
    
    





