clear; 

file_path = 'Fish data/Uturndata2_2fish.csv'; 

data = readmatrix(file_path);

time_interval = 0.02; 
start_row = 1800; 
end_row = 1900; 

% Extract unit vectors for each fish velocity
UX1 = data(start_row:end_row, 11);
UX2 = data(start_row:end_row, 13); 
UY1 = data(start_row:end_row, 12); 
UY2 = data(start_row:end_row, 14); 

time_vector = (1:size(UX2)) * time_interval;

% Calculate the sum of x and y positions for each fish
sum_UXY_fish1 = sum([UX1, UX2], 2);
sum_UXY_fish2 = sum([UY1, UY2], 2);

% Calculate polarization for each time point
P_t_2fish = 1 / 2 * sqrt((sum_UXY_fish1.^2 + sum_UXY_fish2.^2));

% Plot polarization over time
plot(time_vector, P_t_2fish, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Polarization');
title('Polarization Over Time');
grid on;

