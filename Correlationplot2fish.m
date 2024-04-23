clear;

data = readmatrix('Fish data/Uturndata2_2fish.csv');

% Define start and end rows for data subset
start_row = 6340;
end_row = 6540;

% Extract subset of data
subset_data = data(start_row:end_row, :);

% Extract columns corresponding to the position of two fish
e_i = subset_data(:, [11, 12]);
e_j = subset_data(:, [13, 14]); 

% Remove rows with any NaN values
nan_indices_i = any(isnan(e_i), 2);
nan_indices_j = any(isnan(e_j), 2);
e_i = e_i(~nan_indices_i, :);
e_j = e_j(~nan_indices_j, :);

time_delay_range = linspace(2, 0, 100);

H_values = compute_H_values(e_i, e_j, time_delay_range);

% Define time in seconds
time_seconds = (1:size(H_values, 2)) * 0.02;

% Define parameters for C computation
w = 2; % Width of the window
delta_t = 0.02; % Time step

C_values = compute_C_values(H_values, time_seconds, time_delay_range, w, delta_t);

figure;
imagesc(time_seconds, time_delay_range, H_values');
xlabel('Time (s)'); 
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]); % Set color limits
title('Heatmap of H Values for two fish');

figure;
imagesc(time_seconds, time_delay_range, C_values');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('Heatmap of C Values for two fish');

% Function to compute H values
function H_values = compute_H_values(e_i, e_j, time_delay_range)

    min_length = min(size(e_i, 1), size(e_j, 1));
    e_i = e_i(1:min_length, :);
    e_j = e_j(1:min_length, :);

    % Reshape into vectors
    e_i_vector = reshape(e_i, [], 1);
    e_j_vector = reshape(e_j, [], 1);

    H_values = zeros(length(time_delay_range), length(e_i_vector)); 

    % Loop over time delay range to compute H values
    for i = 1:length(time_delay_range)
        tau = time_delay_range(i);
        e_j_tau = e_j_vector - tau;
        H_values(i, :) = e_i_vector .* e_j_tau;
    end
end

% Function to compute C values
function C_values = compute_C_values(H_values, time_seconds, time_delay_range, w, delta_t)

    time_grid = (1:size(H_values, 2)) * delta_t;
    
    % Initialize matrix for C values
    C_values = zeros(length(time_delay_range), length(time_grid));

    % Loop over time delay range and time grid to compute C values
    for i = 1:length(time_delay_range)
        for j = 1:length(time_grid)
            t = time_grid(j);
            % Compute sum of H values within the window
            sum_H = sum(H_values(i, :) .* (abs(time_seconds - t) <= w * delta_t));        
            % Compute C value
            C_values(i, j) = 1 / (2 * w + 1) .* sum_H;
        end
    end
end

