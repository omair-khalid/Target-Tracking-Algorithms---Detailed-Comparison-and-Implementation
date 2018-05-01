% Lab 4  :  Particle Filter (MULTI SESNSOR FUSION AND TRACKING )
% Author :  Wajahat Akhtar, Yu Liu and Omair Khalid

close all;
clear all; 
timestep = 30; 
x0 = [200, 10, 200, 10]'; % set range for state vector ([x, x_speed, y, y_speed])
x0_init = x0 .* rand(4, 1); % target's initial state

sigma_Q = 2;
sigma_R = 2;

delta_t = 1;
% motion model (markov kernel)
F = [1, delta_t, 0, 0;
    0, 1, 0, 0;
    0, 0, 1, delta_t;
    0, 0, 0, 1];

q1 = [(delta_t^4) / 4, (delta_t^3) / 2; (delta_t^3) / 2, delta_t^2];
q2 = [0, 0; 0, 0];
% motion uncertainty
Q = sigma_Q^2 * [q1, q2; q2, q1];

% measurement uncertainty
R = sigma_R^2 * [1, 0; 0, 1];

% map state to measurement 
H = [1, 0, 0, 0; 0, 0, 1, 0];
I = eye(4);
num_gauss = 15; % initial number of gaussian mixure
gauss = cell(num_gauss, 1);
x0_g = [500, 10, 500, 10]';
for i = 1:num_gauss
    gauss{i}.weight = 1 / num_gauss;
    gauss{i}.mean = x0_g .* rand(4, 1);
    gauss{i}.cov = diag( [10, 10, 10, 10]);
end

x_gt = cell(timestep, 1); % ground truth
z = cell(timestep, 1); % measurement/observation
figure, hold on, 
xlim([-100, 500]);
ylim([-100, 500]);
legend('groundtruth', 'measurement', 'tracker');
% writerObj = VideoWriter('gaussia_sum4.avi');
% writerObj.FrameRate = 2; 
% open(writerObj);

% Target trajectory simulator
for k = 1:timestep
    x_gt{k} = mvnrnd(F * x0_init, Q)'; % motion model
    x0_init = x_gt{k};
    z{k} = mvnrnd(H * x_gt{k}, R)'; % measurement model
    plot(x_gt{k}(1), x_gt{k}(3), 'bx', 'LineWidth',2); pause(0.1)
    plot(z{k}(1), z{k}(2), 'g+',  'LineWidth',2); pause(0.1)
    
    wk_sum = 0;
    % Kalman prediction and update
    for j = 1:size(gauss, 1)
        gauss{j}.mean_pred = F* gauss{j}.mean;
        gauss{j}.cov_pred = F*gauss{j}.cov*F' + Q;
        gauss{j}.kalman_g = gauss{j}.cov_pred * H' * (H*gauss{j}.cov_pred*H' + R)^-1;
        gauss{j}.mean_update = gauss{j}.mean_pred + gauss{j}.kalman_g * (z{k}-H * gauss{j}.mean_pred);
        gauss{j}.cov_update = (I - gauss{j}.kalman_g * H) * gauss{j}.cov_pred;
       
        gauss{j}.weight = gauss{i}.weight * mvnpdf(z{k}, H*gauss{j}.mean_update, R + H*gauss{j}.cov_update*H');
        wk_sum = wk_sum + gauss{j}.weight;
    end
    
    x = [];
    y = [];
    % Update Gaussian weight by likelihood and normalize
    for j = 1:size(gauss, 1)
        gauss{j}.weight =  gauss{j}.weight / wk_sum;
        gauss{j}.weight
        gauss{j}.mean = gauss{j}.mean_update;
        gauss{j}.cov = gauss{j}.cov_update;
        x = [x, gauss{j}.mean(1)];
        y = [y, gauss{j}.mean(3)];
    end
    
    h = plot(x, y, 'ro'); pause(0.5);
    legend('groundtruth', 'measurement', 'tracker');
    
    M(k) = getframe;
    
    set(h,'Visible','off');
end

