% This function predicts the output of SVM given model parameters and test
% data

function pred = pred(alpha1, y, train, newpt, b)
	pred = (calc(alpha1, y, train, newpt, b) > 0);
	pred = (pred * 2) - 1;
	pred = pred';
end
