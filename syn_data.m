% This file uses the implementation of TSVM to find the hyperplanes for the
% synthetic 2D dataset and plots the planes, decision boundaries and SVs.

A = [];
B = [];
i = 1;

X = [ LNSX1 LNSY1 ;
      LNSX2 LNSY2 ];
lb1 = ones([size(LNSX1,1),1]);
lb2 = ones([size(LNSX2,1),1]);
x_label = [ lb1 ; lb2.*-1 ];

n = size(X,1);
while i < n
    if x_label(i,1) == 1
        A = [A ; X(i,:)];
    else 
        B = [B ; X(i,:)];
    end
    i = i + 1;
end

n1 = size(A,1);
n2 = size(B,1);

n1_train = round(1.0*n1);
train_A = A(1:n1_train,:);


n2_train = round(1.0*n2);
train_B = B(1:n2_train,:);


C = [ train_A ; train_B] ;
KA = kernel_func(train_A,C);
KB = kernel_func(train_B,C);
e1 = ones([size(train_A,1),1]);
e2 = ones([size(train_B,1),1]);

S = [KA e1];
R = [KB e2];
iden1 = eye(size(S'*S,1));
epsilon1 = 100;
W1 = R*inv(S'*S+epsilon1*iden1)*R';
zero1 = zeros([size(train_B,1),1]);
Cost1 = 1;
cvx_begin
    variable x1(size(train_B,1));
    maximize(e2'*x1 - 0.5*x1'*W1*x1);
    subject to
        zero1 <= x1;
        x1 <= Cost1;
cvx_end

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

iden2 = eye(size(R'*R,1));
epsilon2 = 100;
W2 = S*inv(R'*R+epsilon2*iden2)*S';
zero2 = zeros([size(train_A,1),1]);
Cost2 = 1;
cvx_begin
    variable x2(size(train_A,1));
    maximize(e1'*x2 - 0.5*x2'*W2*x2);
    subject to
        zero2 <= x2;
        x2 <= Cost2;
cvx_end
disp(x2);
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

size(USV1,1)
size(BSV1,1)
size(USV2,1)
size(BSV2,1)

z1 = - (inv(S'*S+epsilon1*iden1)*R'*x1);
u1 = z1(1:size(C,1),:);
b1 = z1(size(C,1)+1:size(C,1)+1,:);

z2 = - (inv(R'*R+epsilon2*iden2)*S'*x2);
u2 = z2(1:size(C,1),:);
b2 = z2(size(C,1)+1:size(C,1)+1,:);

[xg,yg] = meshgrid(-2:.05:2,-2:.05:2);
xg = xg(:);
yg = yg(:);
label = syndata_pred([xg,yg],C,u1,b1,u2,b2);
hold on;
gscatter(xg,yg,label,'bwcg','ssoo');
scatter(LNSX1,LNSY1,'o','b');
scatter(LNSX2,LNSY2,'o','m');
scatter(USV1(:,1),USV1(:,2),100,'d','k','filled');
scatter(USV2(:,1),USV2(:,2),100,'d','r','filled');
legend({'-1','-1','+1','+1','Positive Training Examples','Negative Training Examples','Unbounded Support Vectors for 1','Unbounded Support Vectors for 2'})
hold off;
