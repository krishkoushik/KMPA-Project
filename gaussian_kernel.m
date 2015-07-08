% This file computes the value of rbf kernel for 2 input vectors

function s = gaussian_kernel(x,y,sigma)
    s = exp(-sigma .* pdist2(x,y,'euclidean').^2);
end