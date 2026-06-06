clear; clc;
% 腿长最长为17cm 0.17m，最短为9cm 0.09m
L0s = 0.09:0.005:0.17; % L0变化范围
Ks = zeros(2, 6, length(L0s)); % 存放不同L0对应的K

for step = 1:length(L0s)
    % 所需符号量
    syms theta theta1 theta2; 
    syms x x1 x2;
    syms phi phi1 phi2;
    syms T Tp N P Nm Pm Nf t;
    
    % 机器人结构参数
    R = 0.060; L = L0s(step)/2; Lm = L0s(step)/2; l = 0.02; mw = 0.6; mp = 0.08542; M = 2.491; 
    Iw = 0.5 * mw * R^2; 
    Ip = mp * ((L+Lm)^2) / 3.0; 
    Im = 0.014504;
    g = 9.8;
    
    % 进行物理计算
    Nm = M*(x2+(L+Lm)*(theta2*cos(theta)-theta1^2*sin(theta))-l*(phi2*cos(phi)-phi1^2*sin(phi)));
    Pm = M*g+M*((L+Lm)*(-theta1^2*cos(theta)-theta2*sin(theta))-l*(phi1^2*cos(phi)+phi2*sin(phi)));
    N = Nm+mp*(x2+L*(theta2*cos(theta)-theta1^2*sin(theta)));
    P = Pm+mp*g+mp*L*(-theta1^2*cos(theta)-theta2*sin(theta));
    
    equ1 = x2-(T-N*R)/(Iw/R+mw*R);
    equ2 = (P*L+Pm*Lm)*sin(theta)-(N*L+Nm*Lm)*cos(theta)-T+Tp-Ip*theta2;
    equ3 = Tp+Nm*l*cos(phi)+Pm*l*sin(phi)-Im*phi2;
    [x2, theta2, phi2] = solve(equ1, equ2, equ3, x2, theta2, phi2);

    % 求得雅克比矩阵，然后得到状态空间方程
    Ja = jacobian([theta1; theta2; x1; x2; phi1; phi2],[theta theta1 x x1 phi phi1]); 
    Jb = jacobian([theta1; theta2; x1; x2; phi1; phi2], [T Tp]); 
    
    % 代入平衡点
    A = vpa(subs(Ja, [theta theta1 x x1 phi phi1 T Tp],[0 0 0 0 0 0 0 0]));
    B = vpa(subs(Jb,[theta theta1 x x1 phi phi1 T Tp],[0 0 0 0 0 0 0 0]));
    
    % 离散化与求解LQR
   [G,H] = c2d(eval(A), eval(B), 0.002);
   Q_=diag([500 30 100 20 500 5]);
   R_=diag([10 1]);               %T Tp
    Ks(:,:,step) = dlqr(G, H, Q_, R_);
end

coeffs_table = zeros(12, 4);
idx = 1; 
for y_idx = 1:6       
    for x_idx = 1:2  
        p = polyfit(L0s, reshape(Ks(x_idx, y_idx, :), 1, length(L0s)), 3);
        coeffs_table(idx, :) = p;
        idx = idx + 1;
    end
end

fprintf('float Poly_Coefficient[12][4] = {\n');
for i = 1:12
    fprintf('    {%22.6ff, %22.6ff, %22.6ff, %22.6ff}', ...
            coeffs_table(i,1), coeffs_table(i,2), coeffs_table(i,3), coeffs_table(i,4));
    if i ~= 12
         fprintf(',\n');     
    else
        fprintf('\n');
        fprintf('\n');
    end
end
fprintf('};\n');

fprintf('\n');
fprintf('POLY_COEFFICIENT = np.array([\n');
for i = 1:12
     fprintf('    [ %22.6f, %22.6f, %22.6f, %22.6f],', ...
            coeffs_table(i,1), coeffs_table(i,2), coeffs_table(i,3), coeffs_table(i,4));
    if i ~= 12
        fprintf('\n');  
    else
        fprintf('\n');
        fprintf('\n');
    end
end
fprintf('])\n');