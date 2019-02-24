clear all;
close all;
clc;

RGB = imread('colorful.jpg');
GRAY = rgb2gray(RGB);
figure;
imshow(GRAY,[0,255]);

% grayscale min value
l = 0;
% grayscale max value
h = 255;
% grayscale value range
x = l:1:h;
% the image's probability-density-function
p = imhist(GRAY);
p = p ./ sum(p);

% allocate the errors vector
e = zeros(8,1);
% create a figure with a grid. Use the next outputs to be on the same plot.
figure('Name','q4-3');
hold on;
grid on;
% we need to do max-lloys quantization using bits in [1..8] 
for b = 1:1:8
    % number of intervals that can be represented with b bits
    J = 2^b;
    % length of each interval
    delta = (h - l) / J;
    % initialization of max-lloyd
    d = l:delta:h;
    % to calculate are we put each value d_i above d_(i+1) in a matrix and then
    % we sum up the elements in each columns. Aftwerwards we divide by 2 to get
    % the mid point of each interval [d_i,d_(i+1)]. Lastly we take J first
    % elements since (d_(J)+0) is not a valid r_i value.
    r = (sum([d ; d(2:end) 0])) ./ 2;
    r = r(:,1:(end-1));

    % let us calculate the values in the numerator of the representation's 
    % equation so we can use it later. 
    % the numerator values are just p'.
    xp = (x .* p');

    % we decided to iterate until the absolute difference between the newly
    % calculated r_i values and the previously calculated r_i values are less 
    % then a certain thershold. 
    r_prev = zeros(1, J);
    thresh = 1e-6;

    % we iterate until there is no big of a change in r_i's
    while (max((abs(r - r_prev))) > thresh)
        % these are the newly calculated d_i values. Note that we use here the
        % same "trick" as we used before to calculate the initial r_i values.
        % Afterwards, we add to the vector d_0 and d_J (the first and last
        % elements which are constant)
        d = [l (sum([r(:,1:(end-1)); r(:,2:end)]) ./ 2) h];
        % before changing r, lets save it, so we can later compare in the while condition
        % the new values with the old ones.
        r_prev = r;
        % now we want to calculate the r_i's based on the d_i values. We do
        % that by using the equation we have seen in the tutorial. Instead of
        % the integral we use a sum since the interval here is discrete. For
        % each interval [d_i,d_(i+1)] we sum up al relevant values in the xp
        % and p vectors.
        for i = 1:1:J 
            numerator = sum(xp(:, (ceil(d(i)) + 1):(floor(d(i+1)) + 1)),2);
            denominator = sum(p((ceil(d(i)) + 1):(floor(d(i+1)) + 1), :));
            r(i) = numerator / denominator;
        end
    end
    % after finished calculating the d_i and r_i values which setisfy our
    % threshold condition, plot them out.
    plot(b, d, 'r.');
    plot(b, r, 'b.');
    % and save them in the errors vector (as we did in section 2 of this
    % question)
    e(b) = sum((x - repelem(r, ceil((h+l)/J))).^2 .* p'); 
end
% add some additional descriptions to the plot
title('Decision and Representation Levels for Representative ? Bits Using Max-Lloyd Quantizer');
xlabel('# Bits')
ylabel('Decision & Representation levels')
legend('\color{red} Decision','\color{blue} Representation')
hold off;
% after calculations we see that the mean-square-error (MSE) decreases as we
% increase the number of bits b (as expected). the error is 0 when we have
% enough bits to represent all 256 different values i.e. when b = 8.