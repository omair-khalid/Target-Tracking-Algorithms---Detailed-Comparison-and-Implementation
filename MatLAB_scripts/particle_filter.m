% Lab 4  :  Particle Filter (MULTI SESNSOR FUSION AND TRACKING )
% Author :  Wajahat Akhtar, Alpha Liu and Omair Khalid

clc
clear all ;
close all ;

%% Initialization Particle filter
delta_t = 1 ;           % Frequency = frames per sec ( freqeucy )
Sig_measurement   = 2 ; % Noise variance value for observation noise
Sig_simulator   = 2 ;   % Noise vairance value for measurement noise

sigma_observation = (Sig_measurement)^2 ; % Q matrice observation noise
sigma_simulator = (Sig_simulator)^2 ;     % R matrice measurement noise

% Creating Ground truth constant velocity model
w = [] ;
% Covariance matrix
P = [1000 0 0 0 ; 0 1000 0 0 ; 0 0 1000 0 ; 0 0 0 100 ] ; 
% Identity
I = eye(4);
% State Transition Model
F  = [ [ 1 delta_t ; 0 1] , [0 0 ; 0 0 ] ; [0 0 ; 0 0 ] , [ 1 delta_t ; 0 1] ];
% Process noise Variance
Q = (sigma_simulator)^2 * [ [ ((delta_t)^4 /4)  ((delta_t)^3 /2) ; ((delta_t)^3 /2)  (delta_t)^2 ] , [0 0 ; 0 0 ] ; [0 0 ; 0 0 ] , [ ((delta_t)^4 /4)  ((delta_t)^3 /2) ; ((delta_t)^3 /2)  (delta_t)^2 ] ];
% R matrice 
R = (sigma_observation)^2 * eye(2);
% H matrice
H = [1 0 0 0 ; 0 0 1 0 ] ;


%% Generating particles i to N for a single target
num_targets = 1 ;
P = cell(num_targets,1);
N = 120 ; % number of particles
x_size = 200 ; % intial x position of object
y_size = 200 ; % intial y position of object
x_velocity_std = 10 ; % intializing velocity x of object
y_velocity_std = 10 ; % intializing velocity y of object

particles = [ 200 + 600 ; x_velocity_std ;  200 + 600 ;y_velocity_std ].*rand(4,N) ;
state_g = [ x_size ; x_velocity_std ; y_size;y_velocity_std ].*rand(4,num_targets) ;

ground_truth = cell(num_targets,1) ;
ground_truth_vect = [] ;
measurement = cell(num_targets,1) ;
measurement_vect = [] ;

w_n = [];
w_sum = 0 ;
time_step = 25 ;

%% Plotting particles
% figure;
% hold on
% plot(particles(1,:),particles(3,:),'+r');

%% SIMULATOR
for k = 1 : time_step
    
    for i = 1 : num_targets
        state_g(:,i) = mvnrnd( F * state_g(:,i) , Q )';% Observation
        ground_truth{i} =  state_g(:,i) ;
        
        z_temp = mvnrnd( H*state_g(:,i) , R );% measurement
        measurement{i}  = z_temp' ;
  
    end
    ground_truth_vect = [ground_truth_vect , ground_truth ];
    measurement_vect = [ measurement_vect , measurement ] ;
    
end


simulator_state_gt = []  ;
simulator_measurement_gt = []  ;

%% PLOTTING SIMULATOR VALUES

figure ;
hold on
% Writing Video Script for figures Uncomment if needed 
% writerObj = VideoWriter('particle_filter.avi');
% writerObj.FrameRate = 2; 
% open(writerObj);

for i = 1 : num_targets
    
    simulator_measurement_gt_tmp_x = [];
    simulator_measurement_gt_tmp_y =[];
    simulator_state_gt_tmp_x = [];
    simulator_state_gt_tmp_y = [];
    
    for j = 1: time_step
        simulator_measurement_gt_tmp_x(j,1) = measurement_vect{i,j}(1,1) ;
        simulator_measurement_gt_tmp_y(j,1) = measurement_vect{i,j}(2,1) ;
        simulator_state_gt_tmp_x(j,1) = ground_truth_vect{i,j}(1,1) ;
        simulator_state_gt_tmp_y(j,1) = ground_truth_vect{i,j}(3,1) ;
        
        
        plot(simulator_state_gt_tmp_x(j,1),simulator_state_gt_tmp_y(j,1),'-o');
        plot(simulator_measurement_gt_tmp_x(j,1) , simulator_measurement_gt_tmp_y(j,1),'-x' )
%         legend( 'Ground truth',' Measurements');
%         title(['Simlator for Single object Tracking with Time Step k = '  num2str(j) ]) ;
%         frame = getframe(gcf);
%         writeVideo(writerObj, frame); 
        pause(0.5) 
        
    end 
end

% Closing Video
% close(writerObj);

state_vect = cell(1,time_step) ;
sensor_measure_vect  = cell(1,time_step) ;
state_tmp = [] ;
measure_tmp = [] ;

%% GENRATING PARTICLES

% For i to N Evaluating the importance weight
for i = 1 : N
    w(i) = 1 / N ;
end

% initializing weights
weight_sum = 0 ;
w_covar = 0 ;
w_mean = 0  ;


for k = 1 : time_step
     
    %% Prediction
    particles =  mvnrnd( (F * particles)' , Q )'; % Measurement
    plot(particles(1,:),particles(3,:),'g+') ; 

    %% Estimation
    for i = 1 : N
        w_n(i) =  mvnpdf( measurement_vect{1,k}(:,1), H * particles(:,i), R ) ;
        w_sum = w_sum + w_n(i) ;
    end
    
    %% Normalize the weight
    for i = 1 : N
        w(i) = w_n(i) / w_sum ;
    end
    w_sum = 0 ;
    
    %% RESAMPLING
    % Mean
    for i = 1 : N
        w_mean = w_mean  + ( w(i)* particles(:,i) ) ;   
    end
    
    % Covariance
    for i = 1 : N
        w_covar = w_covar +  w(i) * (w_mean - particles(:,i) ) * (w_mean - particles(:,i) )' ;
    end
    
    % Update
    for i = 1 : N
        updated_particles(:,i) = mvnrnd(w_mean ,w_covar ) ;
    end

    particles = updated_particles ;
    plot(particles(1,:),particles(3,:),'+r');
    legend(' Ground truth',' Measurements ','Particles','Weighted Particles') ;
    w_covar = 0 ;
    w_mean = 0 ;
    
end

disp('End ');



