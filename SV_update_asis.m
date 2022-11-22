% Consider the model:
% yt ~ N(0, exp(ht)),
% ht = (1 - phi) * mu + phi * htm1 + N(0, sig2), 
% h1 ~ N(mu, sig2/(1 - phi^2).
%
% Priors are as follows:
% p(mu) ~ N(mu0, Vmu), e.g. mu0 = 0; Vmu = 10;
% p(phi) ~ N(phi0, Vphi)I(-1,1), e.g. phi0 = 0.95; invVphi = 0.04;
% p(sig2) ~ G(0.5, 2*sig2_s), sig2_s ~ IG(0.5,1/lambda), lambda ~ IG(0.5,1)
%
% Use ASIS to improve estimate of mu, sig2 and h
% Use Horseshoe prior to shrink sig2

function [h, mu, phi, sig, sig2_s, lambda] = SV_update_asis(logy2, h, mu, phi,...
    sig, sig2_s, lambda, prior)
% Inputs:
%   logy2: a T-by-1 vector of log squared returns;
%   h: a T-by-1 vector of log variances to initialize the chain (mean(logy2) for the 1st draw);
%   mu: a scalar of the previous draw of the mean of log variance;
%   phi: a scalar of the previous draw of the AR(1) coef of log variance;
%   sig: a scalar of the previous draw of the signed stdev of log variance;
%   sig2_s: a scalar of the previous draw of the variance of the signed stdev of log variance;
%   lambda: a scalar of the inverse of the scale parameter of the IG prior for sig2_s;
%   prior: a 4-by-1 vector of the prior hyper-parameters in the order of mu0, invVmu, phi0, invVphi.
% Outputs:
%   h: a T-by-1 vector of updated log variances;
%   mu: a scalar of the updated mean of log variance;
%   phi: a scalar of the updated AR(1) coef of log variance;
%   sig2: a scalar of the updated variance of log variance;
%   sig2_s: a scalar of the updated sig2_s;
%   lambda: a scalar of the updated lambda.

%% Preparation
mu0 = prior(1);  invVmu = prior(2); % mean: p(mu) ~ N(mu0, invVmu^(-1)) 
phi0 = prior(3); invVphi = prior(4); % AR(1): p(phi) ~ N(phi0, invVphi^(-1))I(-1,1)
sig2 = sig^2;
T = length(logy2);


%% 10-point normal mixture to approximate log(chi2(1))
pi =   [0.00609  0.04775 0.13057 0.20674  0.22715  0.18842  0.12047  0.05591  0.01575  0.00115];
mi =   [1.92677  1.34744 0.73504 0.02266 -0.85173 -1.97278 -3.46788 -5.55246 -8.68384 -14.65000];  %% means already adjusted!! %%
sigi = [0.11265  0.17788 0.26768 0.40611  0.62699  0.98583  1.57469  2.54498  4.16591  7.33342];
nm = length(pi);


%% Sample S from a 10-point distrete distribution
q = zeros(T,nm);
for j = 1:nm
    q(:,j) = pi(j) * exp(-0.5 * ((logy2 - h - mi(j)).^2) / sigi(j)) / sqrt(sigi(j));
end
q = q./repmat(sum(q,2),1,nm);
temprand = rand(T,1);
S = sum(repmat(temprand,1,nm) > cumsum(q,2),2)+1;
d = mi(S)';
v = 1./sigi(S)';
 

%% Sample hstar using the precision-based algorithm (AA representation)
Hphi = speye(T) - sparse(2:T, 1:(T-1), phi*ones(1,T-1), T, T);
invSigh = sparse(1:T, 1:T, [(1-phi^2); ones(T-1,1)]);
invSigystar = spdiags(v, 0, T, T);
Kh = Hphi' * invSigh * Hphi + sig2 * invSigystar;
Ch = chol(Kh);
hhat = Kh\(sig * invSigystar * (logy2 - d - mu));
hstar = hhat + Ch\randn(T,1);


%% Use hstar to draw mu, sig in the AA representation
% linear regression: logyt2 - mt = mu + sig * ht_star + N(0, sigt)
mixapp_mean = d;
mixapp_stdinv = sqrt(v);
asis_y = (logy2 - mixapp_mean) .* mixapp_stdinv;
asis_x = [mixapp_stdinv  hstar.*mixapp_stdinv];
regest_cov = ([invVmu 0;0 1/sig2_s] + asis_x' * asis_x) \ eye(2);
regest_cov_half = chol(regest_cov)';
regest_mean = regest_cov * ([mu0*invVmu  0]' + asis_x' * asis_y);
regest = regest_mean + regest_cov_half * randn(2,1);
mu = regest(1);
sig = regest(2);
sign_sig = sign(sig);


%% ASIS: compute h
h = mu + sig * hstar;


%% ASIS: sample sig2 based on h
eta1 = (h(1) - mu) * sqrt(1-phi^2);
eta2toT = h(2:T) - phi * h(1:T-1) - mu * (1-phi);
eta = [eta1; eta2toT];
sum_eta2 = sum(eta.^2); 
sig2 = gigrnd(0.5-0.5*T, 1/sig2_s, sum_eta2, 1);
sig = sign_sig * sqrt(sig2);


%% ASIS: sample mu    
Dmu = 1/(invVmu + ((T-1)*((1-phi)^2) + (1-phi^2))/sig2);
muhat = Dmu*(invVmu*mu0 + ...
    h(1)*(1-phi^2)/sig2 + (1-phi)*sum(h(2:T)-phi*h(1:T-1))/sig2);
mu = muhat + sqrt(Dmu)*randn;


%% ASIS: Backout hstar
hstar = (h - mu) / sig;


%% Sample phi based on hstar in the AA representation
Xphi = hstar(1:T-1);
zphi = hstar(2:T);
Dphi = 1/(invVphi + Xphi'*Xphi);
phihat = Dphi*(invVphi*phi0 + Xphi'*zphi);
% phic = phihat + sqrt(Dphi)*randn;
phic = phihat + sqrt(Dphi)*trandn((-1-phihat)/sqrt(Dphi),(1-phihat)/sqrt(Dphi));
if abs(phic)<(1-1e-10)
    phic_logprior = -0.5*((phic-phi0)^2)*invVphi;
    eta = zphi - phic * Xphi;
    eta2 = eta.^2;
    phic_loglike = -0.5*(1-phic^2)*(hstar(1)^2) + 0.5*log(1-phic^2) - 0.5*sum(eta2);
    phic_logprop = -0.5*((phic-phihat)^2)/Dphi; 

    phi_logprior = -0.5*((phi-phi0)^2)*invVphi;
    eta = zphi - phi * Xphi;
    eta2 = eta.^2;
    phi_loglike = -0.5*(1-phi^2)*(hstar(1)^2) + 0.5*log(1-phi^2) - 0.5*sum(eta2);
    phi_logprop = -0.5*((phi-phihat)^2)/Dphi;     
    
    logprob = (phic_logprior + phic_loglike - phic_logprop) - ...
        (phi_logprior + phi_loglike - phi_logprop);
    if log(rand) <= logprob 
        phi = phic;
    end
end 


%% Sample the horseshoe prior parameters for sig2
sig2_s = 1/exprnd(1/lambda+0.5*sig2);
lambda = 1/exprnd(1+1/sig2_s); 


end
