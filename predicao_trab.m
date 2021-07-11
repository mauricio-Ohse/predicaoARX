%%
% Code developed for ARX and ARIMA model estimation. The ARX model is done 
% manually by least squares method while ARIMA model utilizes Box-Jenkins
% built-in matlab function. Done as an grad assignment.
clear all
close all

% load data
data=importdata('trabalhoM')
id_data=iddata(data(:,2),data(:,1))


%% estiimate ARX models with all possible combinations
% Calculo ARX manual
% A0(z)*y=B(z)u+e
% cap 7.1 soderstrom

% Para obtenção da predição por minimos quadrados, monta-se uma matriz phi
% que contem as informações das medidas de y e u atrasadas (de acordo com
% cap 7.1). essa matriz tem a seguinte composição:
% phi'=[-y(t-1) ... -y(t-nf) u(t-1) ... u(t-nb)], onde nf é a ordem do
% polinomio A(z) e nb é a ordem do polinomio B(z). Para cada instante de
% tempo t, existe uma matriz phi.

% A partir de phi e das medidas, obtem-se os parametros estimados por
% theta=sum(phi*phi')^-1*sum(phi*y_medido);
% onde theta=[a_1 a_2 .. a_nf b_1 ... b_nb]'

% Define quais combinações de modelo serão testadas
nf=1:4;
nb=1:4;
NN_man = struc(nf,nb); % struc com todas as possibilidades de nf e nb (respectivamente ordem de A e ordem de B)
N=length(data(:,1));


% Constroi phi
y_medido=data(:,2);
u=data(:,1);

% Variaveis do tipo cell para guardar as informações de cada iteração m
n_models=size(NN_man,1);
theta=cell(n_models);
M=cell(n_models);
y_predicao=cell(n_models);
% errosquare=cell(n_models);

% cada iteração desse loop é um modelo de indice m diferente sendo testado
for m=1:n_models
nf=NN_man(m,1);
nb=NN_man(m,2);
clear phi_col phi sigmaphi sigmaphiy ypred theta_aux

% calcula theta 
ajuste_count=max(1+nf,1+nb);
 for i=1+ajuste_count:N
     % todos os y atrasados
     for j=1:nf
     phi_col(j) = -y_medido(i-j);
     end
     % os u atrasados
     for k=1:nb
     phi_col(j+k) = u(i-k);
     end
     % junta na linha i 
     phi(:,i)=phi_col';
 end

% Metodo dos minimos quadrados revisitado soderstrom cap 7.1
sigmaphi=0;
sigmaphiy=0;
for i=1:N
    sigmaphi=sigmaphi+ phi(:,i)*phi(:,i)';
    sigmaphiy=sigmaphiy+phi(:,i)*y_medido(i);
end

theta_aux=sigmaphi^-1*sigmaphiy;
theta{m}=theta_aux;
M{m}=sigmaphi;

% verifica a qualidade da estimativa
ypred=phi'*theta_aux;
y_predicao{m}=ypred;
errosquare(m)=sum((y_medido-ypred).^2)/N;
end

% Compara os modelos pelo  o criterio de akaike
for m=1:n_models
    n_aux=NN_man(m,:); % n_aux = [na nf] do modelo de indice m
    p=sum(n_aux);
    aicscore(m)=N*log(errosquare(m))+2*p;
end

[best_aic,ind_melhorarx]=min(aicscore);

% Mas qual a qualidade da estimativa feita? em outras palavras, o quão boa
% é a predição dos parametros? em aula vimos as elipses que dão 95% de
% confiabilidade atravez de (theta-theta0)M(theta-theta0)<ksi, com ksi
% vindo da distribuição qui quadrado, M sendo a matriz M definida lá em
% cima, theta o theta real e theta0 o theta estimado.

%procurei no livro do ljung uma explicação e nao achei... mas a equação
%7.91 traz uma relação que acho que sei usar para definir pelo menos a
%variancia dos paramentros: cov (Estimador_theta)>= M^-1, ou seja, se
%inverter a matriz M, a diagonal é a variancia de cada parametro. dai fazer
%um intervalo de 95% de confiança é facil (aprox. dois desvios padrões pra
%cada lado).

cov_modelo=M{ind_melhorarx}^-1;
var_theta=diag(cov_modelo);
best_theta=theta{ind_melhorarx,1};
best_nf=NN_man(ind_melhorarx,1);
best_nb=NN_man(ind_melhorarx,2);


% por fim, obtem a função de transferencia de G e de H. hardcoded pro
% melhor nf=2 e nb=2
B=[best_theta(3) best_theta(4)];
A=[1 best_theta(1) best_theta(2)];
G_est_arx=tf(B,A,1);
H_est_arx=tf(1,A,1);

% ypred_lsim=lsim(G_est_arx,u); %uma simulação apenas com a entrada (como
% leva em desconsideração um ruido filtrado por H, nao tem muita validade)


%% Agora tudo de novo, mas para o BJ
% specify model orders
nf = 1:4;
nb = 1:4;
nd= 0:4;
nk = 0;
nc=1:4;


NN_bj=struc(nb,nc,nd,nf,nk); 
models_bj=cell(size(NN_bj,1),1);

for ct_bj = 1:size(NN_bj,1)
   models_bj{ct_bj}=bj(id_data,NN_bj(ct_bj,:));
end
%% obtem akaike para BJ

V_bj = aic(models_bj{:});
[Vmin_bj,I_bj] = min(V_bj);

%% best model bj
best_model_bj=models_bj{I_bj}
best_NN_bj=[ 4     3     3     3     0]; % hardcodado que foi o melhor obtido

%Tratamento dos dados pro BJ. o modelo é:
%  y(t) = [B(q)/F(q)] u(t-nk) +  [C(q)/D(q)]e(t)
% a função BJ retorna a covariancia dos paramtros
var_BJ=diag(best_model_bj.Report.Parameters.FreeParCovariance);

G_est_bj=tf(best_model_bj.B,best_model_bj.F,1,'Variable','z^-1');
H_est_bj=tf(best_model_bj.C,best_model_bj.D,1,'Variable','z^-1');


%% teste modelo
% esse teste é um teste que não faz mto sentido, visto q temos a entrada u
% e a saida do sistema nos dados, mas não temos nenhuma caracteristica do
% ruído (apenas presumimos ser ruido branco). De qualquer maneira é
% esperado que de pelo menos um pouco parecido.

y_estimado=lsim(G_est_arx,data(:,1));
y_estimadobj=lsim(G_est_bj,data(:,1));
figure
plot(y_estimado,'LineWidth',2,'Color','b')
hold on
plot(y_estimadobj,'LineWidth',2,'Color','r')
plot(data(:,2),'LineWidth',2,'Color','g')
title('Comparação entre dados originais e simulação com G(u)')
xlim([0 200])
grid

figure
plot(y_estimado,'LineWidth',2,'Color','b')
hold on
plot(y_estimadobj,'LineWidth',2,'Color','r')
plot(data(:,2),'LineWidth',2,'Color','g')
title('Zoom comparação entre dados originais e simulação com G(u)')
xlim([0 100])
ylim([3 5.5])
grid

legend('y estimado ARX','y estimado ARIMA','data set')


%% Elipse de intervalo de confiança para parametros ARX  

% separa a covariancia de theta(1) e theta(2)  (ou seja, a1 e a2)
M_aux=M{ind_melhorarx}(1:2,1:2);
P=M_aux^-1; % matriz de covariancia do melhor modelo

% A função abaixo calcula a elipse
% for all theta where: (theta-best_theta)M(theta-best_theta)<ksi
% ksi é o que determina o intervalo de confiança, no caso de 95% é 2.4477
elipses_mod(M_aux,'a1','a2',best_theta(1),best_theta(2),6)
legend('Intervalo de confiança','parametro obtido')

% separa a covariancia de theta(3) e theta(4)  (ou seja, b1 e b2)
M_aux=M{ind_melhorarx}(1:2,1:2);
P=M_aux^-1;

elipses_mod(M_aux,'b1','b2',best_theta(3),best_theta(4),7)
legend('Intervalo de confiança','parametro obtido')
