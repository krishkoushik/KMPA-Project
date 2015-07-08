% function to predict labels for tsvm given paramters of the model and the
% test data
% the plane equation is K(x^T,C)*u + b

function label = tsvmpred(test,C,u1,b1,u2,b2)
i = 1;
label = ones([size(test,1),1]);
while i < size(test,1)
    x = test(i,:);
    Kt = linear_kernel(x,C);
    d1 = abs(Kt*u1 + b1);
    d2 = abs(Kt*u2 + b2);
    if d1 < d2
        label(i,1) = 1;
    else
        label(i,1) = -1;
    end
    i = i + 1;
end