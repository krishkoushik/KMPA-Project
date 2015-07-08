% This file implements twin SVM and measures the performance using
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

accur = ones([1,10]);
time = 0;
for i1=1:10
	train_A = [];
	train_B = [];
	val_A = [];
	val_B = [];
    % splitting training data into training and validation set
	for j=1:10
		if( j == i1)
			val_A = train1(range1(j):range11(j),:);
			val_B = train2(range2(j):range22(j),:);
		else
			train_A = [train_A;train1(range1(j):range11(j),:)];
			train_B = [train_B;train2(range2(j):range22(j),:)];
		end
	end
	%%Access valtraini and valvali here
	C = [ train_A ; train_B] ;
    tic
    
    % computing gram matrix for each class separately
    KA = kernel_func(train_A,C);
    KB = kernel_func(train_B,C);
    
    % setting parameters for the dual optimization function
    e1 = ones([size(train_A,1),1]);
    e2 = ones([size(train_B,1),1]);
    S = [KA e1];
    R = [KB e2];
    
    % epsilon*I is reuqired as S'S may not be invertible
    iden1 = eye(size(S'*S,1));
    epsilon1 = 10;
    W1 = R*inv(S'*S+epsilon1*iden1)*R';
    zero1 = zeros([size(train_B,1),1]);
    Cost1 = 100;
    
    iden2 = eye(size(R'*R,1));
    epsilon2 = 10;
    W2 = S*inv(R'*R+epsilon2*iden2)*S';
    zero2 = zeros([size(train_A,1),1]);
    Cost2 = 100;
    
    % solving for the corresponding dual for the two objective functions
    cvx_begin 
        variable x1(size(train_B,1));
        maximize(e2'*x1 - 0.5*x1'*W1*x1);
        subject to
            zero1 <= x1;
            x1 <= Cost1;
    cvx_end
   
    cvx_begin 
        variable x2(size(train_A,1));
        maximize(e1'*x2 - 0.5*x2'*W2*x2);
        subject to
            zero2 <= x2;
            x2 <= Cost2;
    cvx_end
    
   V% collecting SV statistics for each plane
    BSV1 = [];
    USV1 = [];
    for i=1:size(x1,1)
        if(x1 (i) > 0.000001 && x1 (i) < Cost1 - 0.000001)
            USV1 = [ USV1 ; train_B(i,:) ];
        else 
            if( x1 (i) > 0.000001)
                BSV1 = [ BSV1 ; train_B(i,:) ];
            end
        end
    end
    
    BSV2 = [];
    USV2 = [];
    for i=1:size(x2,1)
        if(x2 (i) > 0.000001 && x2 (i) < Cost2 - 0.000001)
            USV2 = [ USV2 ; train_A(i,:) ];
        else 
            if( x2 (i) > 0.000001)
                BSV2 = [ BSV2 ; train_A(i,:) ];
            end
        end
    end
    size(USV1)
    size(BSV1)
    size(USV2)
    size(BSV2)
    return;
    % compute the parameters of the two hyperplanes
    z1 = - (inv(S'*S+epsilon1*iden1)*R'*x1);
    u1 = z1(1:size(C,1),:);
    b1 = z1(size(C,1)+1:size(C,1)+1,:);

    z2 = - (inv(R'*R+epsilon2*iden2)*S'*x2);
    u2 = z2(1:size(C,1),:);
    b2 = z2(size(C,1)+1:size(C,1)+1,:);
    
    time = time + toc;
   
    % measure the performance on validation set
    val = [val_A ; val_B];
    val_yA = ones([size(val_A,1),1]);
    val_yB = ones([size(val_B,1),1]);
    val_yB = val_yB.*-1;
    ref_label = [ val_yA ; val_yB];
    
    label = tsvmpred(val,C,u1,b1,u2,b2);
    
    [ConfusionMat , labels] = confusionmat( ref_label , label);
    disp(ConfusionMat);
    accur(1,i1) = (ConfusionMat(1,1) + ConfusionMat(2,2))/size(val,1);
	
end
disp(time);
disp(accur);
disp(mean(accur));
disp(std(accur));
