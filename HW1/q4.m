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

figure('Name','q4-2');
hold on;
grid on;
% allocate the errors vector
e = zeros(8,1);
% b is the number of bits
for b = 1:1:8
% number of intervals that can be represented with b bits
J = 2^b;
% length of each interval
delta = (h - l) / J;
% decision levels of uniform quantinizer
d = l:delta:h;
% representation levels of uniform quantinizer
r = (0.5 * delta):delta:((J - 0.5) * delta);
% the mean-squares error.
% we use "repelem" to duplicate the elements in r so that every element in
% x will have a corresponding element which will be  subtracted from it.
% Each x_i that falls in the interval [d_(i-1),d_(i)] will be substacted by
% r_i.
e(b) = sum((x - repelem(r, ceil((h+l)/J))).^2 .* p');

plot(b, d, 'r.');
plot(b, r, 'b.');
end
% add some additional descriptions to the plot
title('Decision and Representation Levels for Representative ? Bits Using Uniform Quantizer');
xlabel('# Bits')
ylabel('Decision & Representation levels')
legend('\color{red} Decision','\color{blue} Representation')
hold off;
