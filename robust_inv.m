% Invert a close-to-sigular matrix by SVD
% Set a floor (e.g. 1e-10) to the small singular values

function c_inv = robust_inv(c)
% Inputs:
%   c: a n-by-n input matrix to be inverted
% Outputs:
%   c_inv: a n-by-n matrix of the inverse of the input matrix

minNum = 1e-100;

n = size(c,1);
if rcond(c) > 1e-15
    c_inv = c\eye(n);
else
    [u,d,v] = svd(c);
    vd = diag(d);
    vd(vd < minNum) = minNum;
    c_inv = v * diag(1./vd) * u';
end

