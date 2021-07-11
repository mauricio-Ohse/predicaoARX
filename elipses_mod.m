% funcao que plota as elipses com intervalo de confianca
% para uma distribuicao normal de duas variaveis theta1 e theta2
% Autor: Binotto
% Referencia: http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
% dados [Mx2] - M = tamanho dos dados

function elipses_mod(M,eixoX,eixoY,Xmed,Ymed,p)
%dados = buffer_thetaN_MQ(:,1:2);
P=M^-1

[v,lambdas] = eig(P) %autovalores / autovetores da P

% verifica qual do autovalores eh maior
[largest_eigenvec_ind_c, r] = find(lambdas == max(max(lambdas)));
largest_eigenvec = v(:, largest_eigenvec_ind_c);

% Obtem o maior e o menor autovalor
largest_eigenval = max(max(lambdas));

if(largest_eigenvec_ind_c == 1)
    smallest_eigenval = max(lambdas(:,2))
    smallest_eigenvec = v(:,2);
else
    smallest_eigenval = max(lambdas(:,1));
    smallest_eigenvec = v(1,:);
end

%angulo do maior eixo da elipse com o eixo X
angle = atan2(largest_eigenvec(2), largest_eigenvec(1));
%converte o angulo para um valor entre 0 e 2pi, pois o original assume
%valores entre -pi e +pi
if(angle < 0)
    angle = angle + 2*pi;
end

mi = [Xmed Ymed];

chisquare_val = 2.4477;
theta_grid = linspace(0,2*pi);
phi = angle;
X0=mi(1);
Y0=mi(2);
a=chisquare_val*sqrt(largest_eigenval);
b=chisquare_val*sqrt(smallest_eigenval);

% elipse em coordenadas x e y
ellipse_x_r  = a*cos( theta_grid );
ellipse_y_r  = b*sin( theta_grid );

%matriz de rotacao
R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

%elipse rotacionada
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;

figure(p);
hold all;
plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'-g','LineWidth',2)

grid on;
hold on;
title('Intervalo de confiança de 95% (elipse)')
% plot(Xmed, Ymed, 'b.','LineWidth',2);
%quiver(X0, Y0, largest_eigenvec(1)*sqrt(largest_eigenval), largest_eigenvec(2)*sqrt(largest_eigenval), '-m', 'LineWidth',2);
%quiver(X0, Y0, smallest_eigenvec(1)*sqrt(smallest_eigenval), smallest_eigenvec(2)*sqrt(smallest_eigenval), '-r', 'LineWidth',2);
scatter(Xmed,Ymed,'r','LineWidth',2)
hold on;
xlabel(eixoX,'interpreter','latex','FontSize',15);
ylabel(eixoY,'interpreter','latex','FontSize',15);
%legend({' correto'},'interpreter','latex','FontSize',15);
end