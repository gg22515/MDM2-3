clear;

data = readmatrix('Fish data/Uturndata_5fish.csv');

% Define start and end rows for data subset
start_row = 2700;
end_row = 2800;

% Extract subset of data
subset_data = data(start_row:end_row, :);

% Extract columns corresponding to the position of fish 1 to 5
e_i = subset_data(:, [18, 19]); 
e_j1 = subset_data(:, [22, 23]); 
e_j2 = subset_data(:, [26, 27]); 
e_j3 = subset_data(:, [30, 31]);
e_j4 = subset_data(:, [34, 35]); 

% Remove rows with any NaN values
nan_indices_i = any(isnan(e_i), 2);
nan_indices_j1 = any(isnan(e_j1), 2);
nan_indices_j2 = any(isnan(e_j2), 2);
nan_indices_j3 = any(isnan(e_j3), 2);
nan_indices_j4 = any(isnan(e_j4), 2);
e_i = e_i(~nan_indices_i, :);
e_j1 = e_j1(~nan_indices_j1, :);
e_j2 = e_j2(~nan_indices_j2, :);
e_j3 = e_j3(~nan_indices_j3, :);
e_j4 = e_j4(~nan_indices_j4, :);

% Define range of time delays
time_delay_range = linspace(2, 0, 100);

% Compute H values for each pair of fish
H_values_1 = compute_H_values(e_i, e_j1, time_delay_range);
H_values_2 = compute_H_values(e_i, e_j2, time_delay_range);
H_values_3 = compute_H_values(e_i, e_j3, time_delay_range);
H_values_4 = compute_H_values(e_i, e_j4, time_delay_range);

% Define time in seconds
time_seconds = (1:size(H_values_1, 2)) * 0.02;

% Define parameters for C computation
w = 0.2; % Width of the window
delta_t = 0.02; % Time step

% Compute C values for each pair of fish
C_values_1 = compute_C_values(H_values_1, time_seconds, time_delay_range, w, delta_t);
C_values_2 = compute_C_values(H_values_2, time_seconds, time_delay_range, w, delta_t);
C_values_3 = compute_C_values(H_values_3, time_seconds, time_delay_range, w, delta_t);
C_values_4 = compute_C_values(H_values_4, time_seconds, time_delay_range, w, delta_t);


figure;
subplot(2, 2, 1);
imagesc(time_seconds, time_delay_range, H_values_1');
xlabel('Time (s)'); 
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('H Values for e_i and e_j1');

subplot(2, 2, 2);
imagesc(time_seconds, time_delay_range, H_values_2');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('H Values for e_i and e_j2');

subplot(2, 2, 3);
imagesc(time_seconds, time_delay_range, H_values_3');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('H Values for e_i and e_j3');

subplot(2, 2, 4);
imagesc(time_seconds, time_delay_range, H_values_4');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('H Values for e_i and e_j4');

figure;
subplot(2, 2, 1);
imagesc(time_seconds, time_delay_range, C_values_1');
xlabel('Time (s)'); 
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('C Values for e_i and e_j1');

subplot(2, 2, 2);
imagesc(time_seconds, time_delay_range, C_values_2');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('C Values for e_i and e_j2');

subplot(2, 2, 3);
imagesc(time_seconds, time_delay_range, C_values_3');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('C Values for e_i and e_j3');

subplot(2, 2, 4);
imagesc(time_seconds, time_delay_range, C_values_4');
xlabel('Time (s)');
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]);
title('C Values for e_i and e_j4');

% Function to compute H values
function H_values = compute_H_values(e_i, e_j, time_delay_range)
    % Ensure both input matrices have the same length
    min_length = min(size(e_i, 1), size(e_j, 1));
    e_i = e_i(1:min_length, :);
    e_j = e_j(1:min_length, :);

    e_i_vector = reshape(e_i, [], 1);
    e_j_vector = reshape(e_j, [], 1);

    H_values = zeros(length(time_delay_range), length(e_i_vector)); 


    for i = 1:length(time_delay_range)
        tau = time_delay_range(i);
        e_j_tau = e_j_vector - tau;
        H_values(i, :) = e_i_vector .* e_j_tau;
    end
end

% Function to compute C values
function C_values = compute_C_values(H_values, time_seconds, time_delay_range, w, delta_t)

    time_grid = (1:size(H_values, 2)) * delta_t;

    C_values = zeros(length(time_delay_range), length(time_grid));

    % Loop over time delay range and time grid to compute C values
    for i = 1:length(time_delay_range)
        for j = 1:length(time_grid)
            t = time_grid(j);

            sum_H = sum(H_values(i, :) .* (abs(time_seconds - t) <= w * delta_t));        

            C_values(i, j) = 1 / (2 * w + 1) .* sum_H;
        end
    end
end

