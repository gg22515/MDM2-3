clear;

file_path = 'Fish data/Uturndata_5fish.csv'; 

data = readmatrix(file_path); 

time_interval = 0.02; 
start_row = 140450;
end_row = 140650; 

% Extract unit vecotrs for each fish velocity
UX1 = data(start_row:end_row, 18); 
UX2 = data(start_row:end_row, 19); 
UY1 = data(start_row:end_row, 22); 
UY2 = data(start_row:end_row, 23); 
UX3 = data(start_row:end_row, 26); 
UY3 = data(start_row:end_row, 27); 
UX4 = data(start_row:end_row, 30); 
UY4 = data(start_row:end_row, 31); 
UX5 = data(start_row:end_row, 34); 
UY5 = data(start_row:end_row, 35);
time_vector = (1:size(UX1)) * time_interval; 

% Calculate the sum of x and y positions for each fish
sum_UXY_fish1 = sum([UX1, UY1], 2);
sum_UXY_fish2 = sum([UX2, UY2], 2);
sum_UXY_fish3 = sum([UX3, UY3], 2);
sum_UXY_fish4 = sum([UX4, UY4], 2);
sum_UXY_fish5 = sum([UX5, UY5], 2);

% Calculate polarization for each time point
P_t_5fish = 1 / 5 * sqrt((sum_UXY_fish1.^2 + sum_UXY_fish2.^2 + sum_UXY_fish3.^2 + sum_UXY_fish4.^2 + sum_UXY_fish5.^2));

% Plot polarization over time for 5 fish
plot(time_vector, P_t_5fish, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Polarization');
title('Polarization Over Time for 5 Fish');
grid on; 
