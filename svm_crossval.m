% This file implements classical SVM and measures the performance using
% standard 10 fold cross validation.

%Reading the dataset the dataset and splitting positive and negative
%examples
data = csvread('liver');
feat = size(data,2);
ionx = data(:,1:feat-1);
iony = data(:,feat);
train1 = [];
train2 = [];
for i=1:size(data,1)
	if(iony(i) == 1)
		train1 = [train1;ionx(i,:)];
	else
		train2 = [train2;ionx(i,:)];
	end
end

% computing the ranges required to do 10 fold cross validation
ot1 = floor(size(train1, 1)/10);
ot2 = floor(size(train2, 1)/10);
range1 = 0:ot1:ot1*9;
range1 = range1+1;
range2 = 0:ot2:ot2*9;
range2 = range2+1;
range11 = ot1:ot1:ot1*10;
range11(10) = size(train1,1);
range22 = ot2:ot2:ot2*10;
range22(10) = size(train2,1);
time = 0;
accur = ones([1,10]);
tic
for i=1:10
	train_A = [];
	train_B = [];
	val_A = [];
	val_B = [];
    
    % splitting training data into training and validation set
	for j=1:10
		if( j == i)
			val_A = train1(range1(j):range11(j),:);
			val_B = train2(range2(j):range22(j),:);
		else
			train_A = [train_A;train1(range1(j):range11(j),:)];
			train_B = [train_B;train2(range2(j):range22(j),:)];
		end
	end
	
	y = [ones(size(train_A,1),1);(-1*ones(size(train_B,1),1))];
    tic
    train = [ train_A ; train_B] ;

    % computing the gram matrix
    K = kernel_func(train,train);
    C = 100;
    zero1 = zeros([size(train,1),1]);
    
    % solving for alpha by maximizing the dual
    cvx_begin
        variable alpha1([size(train,1),1]);
        maximize(sum(alpha1) - 0.5*(alpha1 .* y)' * K * (alpha1 .* y));
        subject to
            zero1 <= alpha1;
            alpha1 <= C;
            alpha1'*y == 0;
    cvx_end
    
    % computing b from support vectors
    b = 0;
    nUBSV = 0;
    nBSV = 0;
    for j=1:size(alpha1)
        if (alpha1 (j) < C - 0.00001 && alpha1 (j) > 0.000001)
            % collecting SV statistics
            b = 1*y(j) - calc(alpha1, y, train, train(j,:), 0);
            nUBSV = nUBSV + 1;
        else
            
            if (alpha1(j) > 0.000001)
                
                nBSV = nBSV + 1;
            end
        end
    end
    time = time + toc;

    % measuring performance on the validation set
    val = [val_A;val_B];
    valy = [ones(size(val_A,1),1);(-1*ones(size(val_B,1),1))];
    predvaly = pred (alpha1, y, train, val, b);
    [ConfusionMat , labels] = confusionmat( valy , predvaly);
    disp(ConfusionMat);
    accur(1,i) = trace(ConfusionMat) * 1.0 / sum(sum(ConfusionMat));

end
disp(time);
disp('Bounded Unbounded-')
disp(nBSV);
disp(nUBSV);
disp(accur);
disp(mean(accur));
disp(std(accur));

