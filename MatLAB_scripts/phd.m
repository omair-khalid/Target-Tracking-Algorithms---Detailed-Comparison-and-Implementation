% Lab 4  :  Particle Filter (MULTI SESNSOR FUSION AND TRACKING )
% Author :  Wajahat Akhtar, Yu Liu and Omair Khalid

close all;
clear all; 

%% param definition
num_target =10; 
timestep = 60; 
x0 = [200, 10, 200, 10]'; % set range for state vector ([x, x_speed, y, y_speed])
x0_fa = [800, 800]'; % set false alarm x and y range
x_false_alarm_all = []; 

x0_all = []; % initialize all targets' state randomly
for i = 1:num_target
    x0_all = [x0_all, x0 .* rand(4, 1)];
end

sigma_Q = 0.5;
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
I = diag([1, 1, 1, 1]);

% area of the surveillance region
vol = x0(1) * x0(3);

% adjustable params for non-ideal tracking  
lambda =1; % false alarm level 
Pd = 0.98; % detection rate (% of detecting an object) 
Ps = 0.99; % survival rate (% of an object to survive in this timestep)

merge_thresh = 4; % U
max_gauss_num = num_target*1; % Jmax

% initializes gaussians mixure
num_gauss = num_target*2; % initial number of gaussian mixure
gauss = cell(num_gauss, 1);
for i = 1:num_gauss
    gauss{i}.weight = 0.5;
    gauss{i}.mean = x0 .* rand(4, 1);
    gauss{i}.cov = diag( [10, 10, 10, 10]);
end

% define new-birth gaussian mixures
num_inject_gauss = ceil(num_target*3);
inject_gauss = cell(num_inject_gauss, 1);
for i = 1:num_inject_gauss
    inject_gauss{i}.weight = 0.5;
    inject_gauss{i}.cov = diag( [10, 10, 10, 10]);
end

lose_track = []; % for checking which objects lose tracking as time goes by
%% trajectory simulator + PHD 
x_gt = cell(timestep, 1); % ground truth
z = cell(timestep, 1); % measurement/observation
figure, hold on, 
xlim([-100, 700]);
ylim([-100, 700]);

% writerObj = VideoWriter('phd6.avi');
% writerObj.FrameRate = 2; 
% open(writerObj);
% main control loop!
for k = 1:timestep
    % data simulation
    num_fase_alarm = poissrnd(lambda); % every timestep, differnt number of false alarm
    x_gt_tem = [];
    z_tem = [];
    
    for j = 1: num_target % each target object follows its own trajectory (groudtruth) and are observed (measurement) accordingly
        if rand < Ps % target survives in this timestep
            x_gt_tem = [x_gt_tem, mvnrnd(F * x0_all(:, j), Q)']; % motion model
        else
            % target dies (equvalent to setting its positions ridiculously high, which is outside the surveillance region)
            x_gt_tem = [x_gt_tem, [-1000; 0; -1000; 0]]; 
            lose_track = [lose_track, j];
            lose_track = unique(lose_track);
            
            % max number of gaussian (tracker) is reduced due to permanent
            % loss of target
            max_gauss_num = num_target -size(lose_track, 2);
        end
        
        % measurement of this target is obtained 
        if rand < Pd
            z_tem = [z_tem, mvnrnd(H * x_gt_tem(:,j), R)']; % meas model; 
        end
    end
    
    % add false alarm into the measurement cell
    x_gt{k} = x_gt_tem;
    z{k} = z_tem;
    x0_all = x_gt{k};
    
    for p = 1:num_fase_alarm
        x_false_alarm_all = [x_false_alarm_all, x0_fa .* rand(2, 1)]; % randomly positioned false alarm % fa = false alarm range
    end
    z{k} = [z{k}, x_false_alarm_all]; % each measurement can track actual object and/or false alarm

    plot (x_gt{k}(1,:), x_gt{k}(3,:), 'bx', 'LineWidth',1); pause(0.05)
    
    % the "if" statement is added becasue sometime at a timestep, there
    % exists no actual measurement nor false alarm (empty cell leads to error)
    if ~isempty(z{k})
        h2 = plot (z{k}(1,:), z{k}(2,:), 'g+', 'LineWidth',1); pause(0.05) % need to watch out when no measurement is obtained
    end
%     set(h2,'Visible','off');

    x_false_alarm_all = [];
  
    % PHD prediction for each already-existing gaussian component
    for j = 1:num_gauss
        gauss{j}.mean = F* gauss{j}.mean; % motion model
        gauss{j}.cov = F*gauss{j}.cov*F' + Q; % measurement model cov
        gauss{j}.weight = Ps * gauss{j}.weight;
    end
   
    % inject new-birth gaussians
    for inj = 1:num_inject_gauss
        inject_gauss{inj}.mean = x0 .* rand(4, 1);
    end
    % as time goes by, increase the range where new-birth gaussians can be spawned as the
    % objects move around 
    x0 = [x0(1)*1.05; x0(2); x0(3)*1.05; x0(4)];
    
    num_gauss = num_gauss + num_inject_gauss;
    gauss = [gauss; inject_gauss]; % concatenate existing and birth gaussians
    
    % construct PHD update components
    for j = 1: num_gauss
        gauss{j}.Hm = H*gauss{j}.mean;
        gauss{j}.sd = R + H*gauss{j}.cov*H';
        gauss{j}.kalmen = gauss{j}.cov * H' * (gauss{j}.sd)^-1;
        gauss{j}.weight_record = gauss{j}.weight; 
        gauss{j}.update_cov = (I - gauss{j}.kalmen * H) * gauss{j}.cov; % store variable for later use
        gauss{j}.cov_record = gauss{j}.cov;
        gauss{j}.mean_record = gauss{j}.mean;
    end
    
    % PHD update (in the case where no target has been detected)
    for j = 1: num_gauss
        gauss{j}.weight = (1-Pd) * gauss{j}.weight_record;
        gauss{j}.cov = gauss{j}.cov_record;
        gauss{j}.mean = gauss{j}.mean_record;
    end
    
    % more gaussian generated per measurement
    for jz = 1:size(z{k}, 2)
        weight_sum = 0;
        for j = 1: num_gauss
            % new gaussians
            gauss{jz*num_gauss + j}.weight = gauss{j}.weight_record * Pd * mvnpdf(z{k}(:,jz), gauss{j}.Hm, gauss{j}.sd); % dont forget * wk-1
            weight_sum = weight_sum + gauss{jz*num_gauss + j}.weight;
            gauss{jz*num_gauss + j}.mean = gauss{j}.mean_record + gauss{j}.kalmen * (z{k}(:,jz) - gauss{j}.Hm);
            gauss{jz*num_gauss + j}.cov = gauss{j}.update_cov;
        end
        for j = 1:num_gauss
            gauss{jz*num_gauss + j}.weight = gauss{jz*num_gauss + j}.weight / (lambda / vol + weight_sum);
        end
    end
    num_gauss = size(gauss, 1);
    
    % pruning to remove low-weight gaussians and merge gaussians that are
    % close together
    
    % delete gaussians with low weight
    low_weight_list = [];
    for j = 1:num_gauss
        gauss{j}.weight
        if gauss{j}.weight <= 10^-4 || isnan(gauss{j}.weight) % remove gaussian weight below 10^-4
            low_weight_list = [low_weight_list; j];
            gauss{j} = [];
        end
    end
    gauss = gauss(~cellfun('isempty',gauss)); 
    num_gauss = size(gauss, 1);
    
    % merge gaussians close to each other
    gauss_merge = {};
    while num_gauss ~= 0
        gauss_weights = [];
        for j = 1:num_gauss
            gauss_weights = [gauss_weights; gauss{j}.weight];
        end
        [~, max_ind] = max(gauss_weights);
        important_mean = gauss{max_ind}.mean;
        merge_list = [];
        
        merge_weight_total = 0;
        % find close gaussians
        for j = 1:num_gauss
            mean_dist = (gauss{j}.mean -important_mean)' * (gauss{j}.cov)^-1 * (gauss{j}.mean -important_mean);
            if mean_dist <= merge_thresh
                merge_list = [merge_list; j];
                merge_weight_total = merge_weight_total + gauss{j}.weight;
            end
        end
        
        % calculate merged weight, mean and cov
        merge_mean_sum = 0;
        for j = 1:size(merge_list, 1)
             merge_mean_sum = merge_mean_sum + gauss{merge_list(j)}.mean * gauss{merge_list(j)}.weight;
        end
        merge_mean = merge_mean_sum / merge_weight_total;
        
        merge_cov_sum = 0;
        for j = 1:size(merge_list, 1)
             mean_diff = merge_mean - gauss{merge_list(j)}.mean;
             merge_cov_sum = merge_cov_sum + gauss{merge_list(j)}.weight * (gauss{merge_list(j)}.cov + mean_diff * mean_diff');
             gauss{merge_list(j)} = [];
        end
        merge_cov = merge_cov_sum / merge_weight_total;
        
        merged_gauss = struct('weight',merge_weight_total, 'mean', merge_mean, 'cov', merge_cov);
        gauss_merge = [gauss_merge; merged_gauss];
        gauss = gauss(~cellfun('isempty',gauss)); 
%         disp('number of gauss left');
        num_gauss = num_gauss - size(merge_list, 1);
    end
    
    gauss = gauss_merge;
    
    disp('after pruning and merging:')
    num_gauss = size(gauss, 1)
    % final pruning, by only keeping the largest-weight N number of
    % gaussians
    gauss_weights = [];
    final_gauss = {};
    if num_gauss > max_gauss_num
        for j = 1:num_gauss
            gauss_weights = [gauss_weights; gauss{j}.weight];
        end
        [~, max_ind] = sort(gauss_weights, 'descend');
        for j = 1: max_gauss_num
            final_gauss = [final_gauss; gauss{max_ind(j)}];
        end
    end
    gauss = final_gauss;
    disp('after choosing the highest weight gaussian:')
    num_gauss = size(gauss, 1)
    
    timer = 10;
    x = []; % to store all remaining gaussians' x component (display purpose)
    y = []; % to store all remaining gaussians' y component (display purpose)
    
    for j = 1:num_gauss
        x = [x; gauss{j}.mean(1)];
        y = [y; gauss{j}.mean(3)];
        
%         timer = timer -1;
%         if timer == 0
%             set(h3,'Visible','off');
%             timer = 10;
%         end
    end
    h3 = plot (x, y, 'ro'); pause(0.2)
    legend('groundtruth', 'observation', 'tracker');
%     frame = getframe(gcf);
%     writeVideo(writerObj, frame);
    set(h3,'Visible','off');
end

% close(writerObj);

