dataset = '/tmp2/zeta_calc';
addpath('/home/jimwang/libsvm-3.17/matlab');
[y x] = libsvmread(dataset);
lab1 = 1;
lab2 = -1;

block = 100;

d = zeros(10,1);

fea = size(x,2);
%expands pseudo features
for i = 1:10
	starts =  (i-1)*block;
	ends  =   i*block;

	xx = zeros( block, fea*4);
	for j = 1:4
		peg1 = (j-1)*block/4 + 1;
		peg2 =  j*block/4;
		xx(peg1:peg2, (j-1)*fea+1:j*fea)   = x(peg1+starts:peg2+starts,:); 
	end	

	tmp = eigs( xx'*xx );


	d(i) = tmp(1);
end

mean(d)
