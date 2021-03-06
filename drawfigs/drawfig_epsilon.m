%the first column records total running time at each recorded iteration
%the second colum records testing accuracy at each recorded iteratoin
%the second colum records trainig accuracy at each recorded iteratoin
%pure   = csvread('epsilon_ori.txt');
%tw_1 = csvread('./repeat/epsilon_tw4.txt');
%tw_1 = csvread('./epsilon_tw.txt');
%icdm = csvread('epsilon_icdm.txt');
pure   = csvread('./repeat/epsilon_orif.txt');
tw_1 = csvread('./repeat/epsilon_twr.txt');
icdm = csvread('./repeat/epsilon_icdmf.txt');
psd = csvread('./repeat/epsilon_psd.txt');

pure(:,3) = 100.*pure(:,3);
tw_1(:,3) = 100.*tw_1(:,3);
icdm(:,3) = 100.*icdm(:,3);
psd(:,3) = 100.*psd(:,3);

low  = min(tw_1(:,3))-0.1;
high = max( [max(tw_1(:,3)),max(pure(:,3)),max(icdm(:,3))])+0.1;
figure;
%plot training obj versus time
colors = [0 0 0; 1 0.5 0.5; 1 0.5 0; 0.5 0.5 0.5; 0.5 0.5 1];
markers = {'s','x','p','*','d','^'};
markerSpacing = [1 1; 1 1 ; 1 1; 1 1; 1 1];
names = {'ADMM','SDCA-ADMM (this work)','ADMM-S','P-SDCA'};
lineStyles = {'-'};

options.legendLoc = 'SouthEast';
options.logScale = 0;
options.colors = colors;
options.lineStyles = lineStyles;
options.markers = markers;
options.markerSize = 12;
options.markerSpacing = markerSpacing;
options.legend = names;
options.ylabel = 'Testing Accuracy';
options.xlabel = 'Time (s)';
options.xlimits = [ 40 180];
options.ylimits = [ 88 high];
xx{1} = pure(:,1)';
xx{2} = tw_1(:,1)';
xx{3} = icdm(:,1)';
xx{4} = psd(:,1)';

yy{1} = pure(:,2)';
yy{2} = tw_1(:,2)';
yy{3} = icdm(:,2)';
yy{4} = psd(:,2)';

prettyPlot( xx, yy, options);
print('epsilon_ts_acc','-depsc2','-r300');
system('gs -o -q -sDEVICE=png256 -dEPSCrop -r300 -o epsilon_ts_acc.png epsilon_ts_acc.eps');

figure;
options.ylabel = 'Training Accuracy';
options.xlabel = 'Time (s)';
xx{1} = pure(:,1)';
xx{2} = tw_1(:,1)';
xx{3} = icdm(:,1)';
xx{4} = psd(:,1)';

yy{1} = pure(:,3)';
yy{2} = tw_1(:,3)';
yy{3} = icdm(:,3)';
yy{4} = psd(:,3)';

prettyPlot( xx, yy, options);
print('epsilon_tr_acc','-depsc2','-r300');
system('gs -o -q -sDEVICE=png256 -dEPSCrop -r300 -o epsilon_tr_acc.png epsilon_tr_acc.eps');

