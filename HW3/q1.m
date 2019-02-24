% Numerically calculate phi's coefficients

dt = 0.001;
t = 0:dt:1;
t1 = [ones(1,250) zeros(1,751)];
t2 = [zeros(1,250) ones(1,500) zeros(1,251)];
t3 = [zeros(1,750) ones(1,251)];
phi = ((4 * t) .* t1) + ((1) .* t2) + ((1 - 4 * (t - 0.75)) .* t3);

phi_w = [0 0 0 0];
phi_w(1) = sum(phi .* 1 .* dt);
phi_w(2) = sum(phi(1:501) .* 1 .* dt) - sum(phi(501:1001) .* 1 .* dt);
phi_w(3) = sum(phi(1:251) .* 1 .* dt) - sum(phi(251:751) .* 1 .* dt) + sum(phi(751:1001) .* 1 .* dt);
phi_w(4) = sum(phi(1:251) .* 1 .* dt) - sum(phi(251:501) .* 1 .* dt) + sum(phi(501:751) .* 1 .* dt) - sum(phi(751:1001) .* 1 .* dt);