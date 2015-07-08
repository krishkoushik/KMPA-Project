% This function is used to calculate the kernel gram matrix using linear
% kernel, for the given set of datapoints.

function K = linear_kernel(A,C)
    i = 1;
    K = ones([size(A,1),size(C,1)]);
    while i <= size(A,1)
        j = 1;
        while j <= size(C,1)
            k = A(i,:)*C(j,:)';
            K(i,j) = k;
            j = j + 1;
        end
        i =  i + 1;
    end
end