% Helper function for SVM
function res = calc(alpha1, y, train, newpt, b)
	k = kernel_func(train,newpt);
	res = ((alpha1 .* y)' * k) + b;
end
