clear all;
close all;
clc;
%% -- section 1 -- %%

% - part b - %

% we define the granularity of the function samples. since we want to
% simulate a continues function we choose the delta between each sample to
% be very small (uniform distribution).
continuous_approx_delta = 0.001;
% calculate the 2D grid of a [0,1]x[0,1] rectangle.
grid = 0 : continuous_approx_delta : 1;
[x_grid,y_grid] = meshgrid(grid,grid);

% define the function's parameters.
omega_x = 5;
omega_y = 2;
A = 5000;
% since cos(x) gets 1 and -1 in the defined box, the highest and lowest
% points are the amplitude and (-amplitude) respectively.
phi_h = A;
phi_l = -A;
% calculate the function's values at the given sampling points (which,
% again simulates a continuous function).
phi_xy=A*cos(2*pi*omega_x*(x_grid)).* cos(2*pi* omega_y *(y_grid));

% plot the function as a greyscale image histogram.
imshow(phi_xy,[phi_l phi_h]);
title('phi_xy histogram');

% figure('Name','phi_xy(continuous)');
% mesh_grid = surf(x_grid,y_grid,phi_xy);
% set(mesh_grid,'LineStyle','none');

% - part c - %
% calculate phi-low and phi-high by finding the the min element and max
% element of the given phi matrix
phi_l = min(min(phi_xy));
phi_h = max(max(phi_xy));

% now we want to numericaly calculate the derivative of phi according to x
% and the derivative of phi according to y of every existing point
% in the defined area. for that we will use the definition of the
% derivative. first we are calculating the value of the function at a close
% point (x+delta or y+delta).
phi_x_delta_y = A*cos(2*pi*omega_x*(x_grid+continuous_approx_delta)).* cos(2*pi* omega_y *(y_grid));
phi_x_y_delta = A*cos(2*pi*omega_x*(x_grid)).* cos(2*pi* omega_y *(y_grid+continuous_approx_delta));
% afterwards, we use the previous calculation to calculate an approximation
% of the derivative in every point.
dev_phi_x = (phi_x_delta_y-phi_xy)./(continuous_approx_delta);
dev_phi_y = (phi_x_y_delta-phi_xy)./(continuous_approx_delta);
%lastly, we are doing an approximation of the integral of the energy of the
% derivative of phi according to x and according to y. the approximation is
% done by creating a dense grid of the total area and calculating the
% sum of the areas of the 3D squares created beneath the energies functions
energy_dev_phi_x = sum(sum((dev_phi_x.^2).*((continuous_approx_delta)^2)));
energy_dev_phi_y = sum(sum((dev_phi_y.^2).*((continuous_approx_delta)^2)));

% the results we are getting are:
% phi_l = 5000
% phi_h = -5000
% energy_dev_phi_x = 6.180334468832723e+09
% energy_dev_phi_y = 9.889214252841669e+08


% - part d - %

% in this part, we need to find N_x,N_y & b which setisfy the non-linear
% optimization problem:
% min {MSE_total(N_x,N_y,b)}
% s.t. N_x*N_y*b = B
%
% we need to do it for 2 different values of B.
B_l = 5000;
B_h = 50000;
% let's define the MSE function.
% the function we have has a uniform partition of values across [0,1]x[0,1]
% so we can use the MSE equation we have seen in class. this equation
% assumes uniform sampling and quantizing. Note that x(1) represents N_x,
% x(2) represents N_y and x(3) represents b.
mse = @(x)(1/(12*(x(1))^2)*energy_dev_phi_x) + ...
    (1/(12*(x(2))^2)*energy_dev_phi_y) + ...
    ((phi_h-phi_l)^2)/(12*(2^(2*x(3))));
% define the starting values for x (for finiding the minimum) for the
% numerical calculation. we set them != 0 so fmincon won't fail on division
% by 0.
x0 = [0.001 0.001 0.001];
% define the constraint for B_l. we have no inequality constraint, only
% equality constraint: N_x*N_y*Nb - B = 0.
% we found out that calculating the minimum mse using the constraint
% B_x*B_y*b <= B is much faster using fmincon. since, in this case,
% changing the contraint from = to <= leads to an equal optimization
% problem (since the mse is built in such a way that will always want as
% big N_x,N_y,b as we may have) we changed the constraint.
con_l = @(x)deal((x(1)*x(2)*x(3)-B_l), []);
% define lower and upper bounds for the values of N_x, N_y, b.
lb_l = [0.001 0.001 0.001];
ub_l = [B_l B_l B_l];
% we need to set options to increase max number of iterations.
options = optimoptions('fmincon','MaxFunctionEvaluations', inf, 'MaxIterations', inf);
[opt_l,f_opt_l] = fmincon(mse,x0,[],[],[],[],lb_l,ub_l,con_l,options);
% in this subsection we do not need to transform N_x,N_y & b to integers.
% the results for B_low:
% N_x = 59.2081
% N_y = 23.6840
% b   = 3.5656
% MSE = 3.5328e+05

% ...and do the same with B_h.
con_h = @(x)deal((x(1)*x(2)*x(3)-B_h), []);
lb_h = [0.001 0.001 0.001];
ub_h = [B_h B_h B_h];
[opt_h,f_opt_h] = fmincon(mse,x0,[],[],[],[],lb_h,ub_h,con_h,options);
% the results for B_high:
% N_x = 154.6462
% N_y = 61.8606
% b   = 5.2266
% MSE = 4.9015e+04
% we can see that these results are very close to the result we got in the
% previous part!

% - part e - %

% in this part we need to find the optimal N_x,N_y,b practically. To do so
% we will use a for loop that tests different combinations of N_x,N_y,b for
% finding minimum MSE while still statisfying the constraints.

% initialize the opt MSE value and the solutions vector.
practical_opt_l = [0 0 0];
practical_opt_val_l = inf;
% now we want to test different combinations of INTEGER N_x,N_y,b values
% for finding optimality. The for loop is running on N_x and N_y, and b is
% derived from them. to keep b an integer and still statisfy the constraint
% N_x*N_y*b <= B we use floor on b's value.
% we are going to save the new matrix we are creating here. for allocate it
% beforehand for performance.
reconstructed_image_l = zeros(1/continuous_approx_delta+1);
best_reconstructed_image_l = zeros(1/continuous_approx_delta+1);
% this variable is going to be explained soon
last_y = 0;
% NOTE: this loop might take about a minutes to complete.
tic;
for N_x = B_l:-1:1
    % now we are going to do a trick in order to speed up the calculation.
    % notice that we are running the loop from the highest value of N_x and
    % backwards. Let us notice the following facts:
    % 1. The maximum value of N_y is floor(B_l/N_x) since N_x*N_y <=B
    % 2. if maximum value of N_y is the same as in previous iteration, the
    %    possible values of b are going to be the same as in previous
    %    iteration.
    % 3. since we only decrease N_x and N_y in each iteration, then if the
    %    max value of N_y is the same as in previous iteration, then we can
    %    not get a better resoult for N_x,N_y,b combination then what we've
    %    got in previous iteration (since b gets the same values but we
    %    decreased N_x / N_y).

    % check if we can generate new values of N_y in this iteration and thus
    % generate new values of b. if not we can skip this iteration.
    if(floor(B_l/N_x) == last_y)
        continue;
    end;
    % calculate the vector of all possible values of N_y that gives maximum
    % N_y for particular b.
    %N_y_values = unique(floor((B_l./(N_x*(1:floor(B_l/N_x))))));

    % and save the new max N_y value.
    last_y = floor(B_l/N_x);
    % this variable is going to be used to save the last b found.
    last_b = 0;
    % now loop through N_y values. notice that we are going backwards
    for N_y = floor(B_l/N_x):-1:1
        b = floor(B_l/(N_x*N_y));
        % if the b that was found is similar to the privous b, then it
        % means that we found a combination of N_x,N_y,b that has lower N_y
        % than the previous iteration (but same b), this means that we are
        % going to get a worse MSE for sure. If this is the case skip this
        % iteration
        if (b == last_b)
          continue;
        end
        last_b = b;
        % As we assumed a uniform quantizer when calculating optimal N_x,N_y,b,
        % each decision level should have the same length. let us calculate the
        % length of each decision level - ((phi_high - phi_low) / N)
        decision_region_length = max((2 * abs(A)) / (2^b),1);
        % now we are going to loop through each square in the grid created by N_x &
        % N_y, find the sampling point which is the avarage value of phi in every
        % square and then quantize the average value we found.
        for i = 0:1:(N_y-1)
            % calculate the starting and ending element of the y-axis (in the phi-
            % matrix) as a function of the current i iteration.
            y_start = ceil(i/(continuous_approx_delta*N_y))+1;
            y_end = floor((i+1)/(continuous_approx_delta*N_y))+1;

            for j = 0:1:(N_x-1)
                % similarly, calculate the starting and ending element of the x-axis
                % (in the phi-matrix) as a function of the current j iteration.
                x_start = ceil(j/(continuous_approx_delta*N_x))+1;
                x_end = floor((j+1)/(continuous_approx_delta*N_x))+1;

                % now that we know the range of elements in the current region, we
                % can create a smaller matrix containing only these elements.
                partial = phi_xy([y_start:y_end],[x_start:x_end]);
                % using a numeric integration (approximation) as we did in part c
                % and then dividing by the square size to find the optimal phi (as
                % we saw in class - for a uniform partition, the average point in
                % each sqaure gives the optimal sampling point).
                phi_opt = (N_x*N_y)*(continuous_approx_delta^2 * sum(sum(partial)));
                % the next calculation finds the quantized value of phi_opt (in the
                % given square). of course, as we can see in the below calculation,
                % the representation level of a uniform quantizer is the mid point
                % of the decision interval.
                quantized_phi_opt = (floor(phi_opt/decision_region_length)    + ...
                    ceil((phi_opt+1)/decision_region_length)) * ...
                    decision_region_length / 2;
                % finally save that value in the adjacent sub-matrix of the final-
                % result-matrix.
                reconstructed_image_l([y_start:y_end], [x_start:x_end]) = quantized_phi_opt;
            end
        end

        % calculate mse with the given N_x Ny & b
        parctical_mse = continuous_approx_delta^2 * sum(sum((phi_xy - reconstructed_image_l).^2));
        % if we get a "better" result for the mse then save the new optimal
        % value and optimal solution.
        if (practical_opt_val_l > parctical_mse)
            practical_opt_val_l = parctical_mse;
            practical_opt_l = [N_x N_y b];
            best_reconstructed_image_l = reconstructed_image_l;
        end
    end
end
toc;
% results for the B_low practical solution:
% N_x = 54
% N_y = 23
% b   = 4
% MSE = 3.5783e+05

% ...and do the same with B_h.
practical_opt_h = [0 0 0];
practical_opt_val_h = inf;
reconstructed_image_h = zeros(1/continuous_approx_delta+1);
best_reconstructed_image_h = zeros(1/continuous_approx_delta+1);
max_N_y = 0;
% NOTE: this run might take about 10 minutes.
tic;
for N_x = B_h:-1:1
    N_y_values = unique(floor((B_h./(N_x*(1:floor(B_h/N_x))))));
    if(N_y_values(end) == max_N_y)
        continue;
    end;
    max_N_y = N_y_values(end);
    for N_y = N_y_values
        b = floor(B_h/(N_x*N_y));

        decision_region_length = (2 * abs(A)) / (2^b);
        for i = 0:1:(N_x-1)
            x_start = ceil(i/(continuous_approx_delta*N_x))+1;
            x_end = floor((i+1)/(continuous_approx_delta*N_x))+1;
            for j = 0:1:(N_y-1)
                y_start = ceil(j/(continuous_approx_delta*N_y))+1;
                y_end = floor((j+1)/(continuous_approx_delta*N_y))+1;
                partial = phi_xy([y_start:y_end], [x_start:x_end]);
                phi_opt = (N_x*N_y)*(continuous_approx_delta^2 * sum(sum(partial)));
                quantized_phi_opt = (floor(phi_opt/decision_region_length)    + ...
                                    ceil((phi_opt+1)/decision_region_length)) * ...
                                    decision_region_length / 2;
                reconstructed_image_h([y_start:y_end], [x_start:x_end]) = quantized_phi_opt;
            end
        end

        % calculate mse with the given N_x Ny & b
        parctical_mse = continuous_approx_delta^2 * sum(sum((phi_xy - reconstructed_image_h).^2));
        % if we get a "better" result for the mse then save the new optimal
        % value and optimal solution.
        if (practical_opt_val_h > parctical_mse)
            practical_opt_val_h = parctical_mse;
            practical_opt_h = [N_x N_y b];
            best_reconstructed_image_h = reconstructed_image_h;
        end
    end
end
toc;
% results for the B_high practical solution:
% N_x = 167
% N_y = 59
% b   = 5
% MSE = 5.7391e+04
% we can see that, also here, the results are pretty close to the result
% we got in the previous part!

% plot the graph of phi after sampling and quantization using bit badget of
% 5000
imshow(best_reconstructed_image_l,[phi_l phi_h]);
% figure;
% mesh_grid = surf(x_grid,y_grid,reconstructed_image_l);
title('Graph of phi_x_y after sampling and quantization using bit badget of 5,000');
% xlabel('x');
% ylabel('y');
% zlabel('phi');
% set(mesh_grid,'LineStyle','none');


% ...now we can do the same for B_high
% plot the graph of phi after sampling and quantization using bit badget of
% 50000
imshow(best_reconstructed_image_h,[phi_l phi_h]);
% figure;
% mesh_grid = surf(x_grid,y_grid,reconstructed_image_h);
title('Graph of phi_x_y after sampling and quantization using bit badget of 50,000');
% xlabel('x');
% ylabel('y');
% zlabel('phi');
% set(mesh_grid,'LineStyle','none');
