% Use SVD and QR to perform the Cholesky decomposition of a symmetric matrix
% Applicable for non-invertible matrices

function C = robust_chol(CC)
% Inputs:
%   CC: a m-by-m symmetric matrix (could be non-invertible)
% Outputs:
%   C: a m-by-m lower triangular matrix such that C * C' = CC

[U,D,~] = svd(CC);
D_half = diag(sqrt(diag(D)));
[~,R] = qr(D_half * U');
C = R';