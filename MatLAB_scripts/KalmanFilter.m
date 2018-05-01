% Lab 1  :  Particle Filter (MULTI SESNSOR FUSION AND TRACKING )
% Author :  Wajahat Akhtar, Yu Liu and Omair Khalid

%Lab 1 - Multi Sensor Fusion and Tracking
%Date - 02/10/2017
clear all
close all
%-------------------------
%Initialization
pause(3);
sigmaQ = 1.2;%std of the Process Noise
sigmaP = 1.2;%std of the Prior at iteration zero
sigmaR = 1.2;%std of the Observation Noise

deltaT = 1;

%State Transition Matrix
f = [1 deltaT;0 1];
F = [f zeros(2,2); zeros(2,2) f];

%Process Noise Covariance matrix
q = [(deltaT^4)/4 (deltaT^3)/2;(deltaT^3)/2 (deltaT^2)];
Q = sigmaQ^2.*[q zeros(2,2); zeros(2,2) q];

%Covariance of the Prior at iteration zero
P = 1000*eye(4,4);

%Mapping of the state on to the measurement space
H = [1 0 0 0; 0 0 1 0];

%Observation Noise Covariance matrix
R = (sigmaR^2).*[1 0;0 1];

%-------------------------
%Simulator

x0 = [0;0;0;0];
numP = 50;

for i = 1:numP

x = mvnrnd(F*x0,Q);
X(:,i) = x';
z = mvnrnd(H*x',R);
Z(:,i) = z';
x0 = x';
end

plot(X(1,:),X(3,:),'bx');
hold on;
plot(Z(1,:),Z(2,:),'g+');
hold on;
%-----------------------
pause(3);
X_new = [];
x0 = [10 20 0.23 49]';
for j=1:numP
   %Construction of Kalman Update
    m = F*x0;
    P = F*P*F' + Q;
    
    K = (P*H')*inv(H*P*H' +R);
    P_ = (eye(4,4) - K*H)*P;
    m_ = m + K*(Z(:,j) - H*m);
    
    %Update
    
    
    x0 = m_;  
    P = P_;
    %str = sprintf('Kalman Filter Tracking %j',variable);
    X_new = [X_new x0];
    
    plot(x0(1,:),x0(3,:),'r*');
    
    %title(str);
    legend('Simulator','Measurement', 'Kalman Update')
    title( ['Kalman Filter Tracking - Time Step = ' num2str( j )] )
    
    hold on;
    
    pause(0.25)


end


