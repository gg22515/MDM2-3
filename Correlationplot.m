data = readmatrix('Fish data/Uturndata_2fish.csv');

start_row = 635;
end_row = 1000;
subset_data = data(start_row:end_row, :);

e_i = subset_data(:, 11);
e_j = subset_data(:, 13);

time_delay_range = linspace(-0.5, 2, 100);
H_values = zeros(length(time_delay_range), length(e_i));

for i = 1:length(time_delay_range)
    tau = time_delay_range(i);
    e_j_tau = e_j - tau; 
    H_values(i, :) = e_i .* e_j_tau;
end

time_seconds = (1:size(H_values, 2)) * 0.02;

imagesc(time_seconds, time_delay_range, H_values');
xlabel('Time (s)'); 
ylabel('Time Delay (s)');
colorbar;
clim([-1 1]); 
title('Heatmap of H Values for two fish');



















