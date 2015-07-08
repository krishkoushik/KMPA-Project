% Label Prediction function for synthetic data which distinguishes between
% points lying to the right and left of the hyperplane

function label = syndata_pred(test,C,u1,b1,u2,b2)
i = 1;
label = ones([size(test,1),1]);
while i < size(test,1)
    x = test(i,:);
    Kt = kernel_func(x,C);
    d1 = Kt*u1 + b1;
    d2 = Kt*u2 + b2;
    dc = abs(d1);
    dd = abs(d2);
    if dc < dd
        if d1 < 0
            label(i,1) = 1;
        else
            label(i,1) = 2;
        end
    else
        if d2 < 0
            label(i,1) = -1;
        else
            label(i,1) = -2;
        end
    end
    i = i + 1;
end