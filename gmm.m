% Script to run GMM on datasets. To run on different datasets, add the name of the dataset in the first line of the code.

% Running for liver dataset
ion = csvread('liver');
feat = size(ion,2);
ionx = ion(:,1:feat-1);
iony = ion(:,feat);

% Separating the 2 classes
train1 = [];
train2 = [];
for i=1:size(ion,1)
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
range11;
range22 = ot2:ot2:ot2*10;
range22(10) = size(train2,1);
range22;
time = 0;
accur = ones([1,10]);
tic;
size(train1);

% To find average accuracy and its standard deviation over all runs in the 10-fold cross validation
avg = 0;
varavg = 0;

% Code for 10-fold cross validation
for i=1:10
	train_A = [];
	train_B = [];
	val_A = [];
	val_B = [];
    
    % splitting training data into training and validation set using the ranges found earlier
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

	train11 = train_A;
	train22 = train_B;
	val1 = val_A;
	val2 = val_B;

	% Training the GMM model for both classes
	obj1 = fitgmdist(train11,3,'CovType','diagonal','Regularize',0.1)
	obj2 = fitgmdist(train22,3,'CovType','diagonal','Regularize',0.1)

	testX = [val1; val2];

	proba1 = pdf(obj1 , testX);
	proba2 = pdf(obj2 , testX);

	pred_proba = [ proba1' ;
				  proba2' ;
				  ];

	C1 = ones([1,numel(val1(:,1))]);
	C2 = ones([1,numel(val2(:,1))]);

	C2 = C2.*2;

	valtest = [ C1 C2 ];          
			 
	[MaxVal,Predictions] = max(pred_proba);
	[ConfusionMat , labels] = confusionmat( valtest , Predictions');
	cmat = ConfusionMat;
	acc = (cmat(1,1) + cmat(2,2)) / (cmat(1,1) + cmat(2,2) + cmat(1,2) + cmat(2,1));
	avg = avg + acc;
	acc
	varavg = varavg + acc*acc;

end

% Printing the accuracy values
avg = avg / 10
varavg = varavg / 10;
sqrt(varavg - avg*avg)
