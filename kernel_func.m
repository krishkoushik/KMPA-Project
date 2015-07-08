% This function is used to calculate the kernel gram matrix using gaussian
% kernel, for the given set of datapoints.

function K = kernel_func(A,C)
    i = 1;
    K = ones([size(A,1),size(C,1)]);
    sigma = 0.0001;
    while i <= size(A,1)
        j = 1;
        while j <= size(C,1)
            k = gaussian_kernel(A(i,:),C(j,:),sigma);
            K(i,j) = k;
            j = j + 1;
        end
        disp(i);
        i =  i + 1;
    end
end