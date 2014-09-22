%the first column records total running time at each recorded iteration
%the second colum records testing accuracy at each recorded iteratoin
%the second colum records trainig accuracy at each recorded iteratoin
tw_0 = csvread('./processed/rcv1.r0.txt');
tw_1 = csvread('./processed/rcv1.r1.txt');
tw_2 = csvread('./processed/rcv1.r2.txt');
tw_3 = csvread('./processed/rcv1.r3.txt');
tw_4 = csvread('./processed/rcv1.r4.txt');
tw_5 = csvread('./processed/rcv1.r5.txt');
tw_6 = csvread('./processed/rcv1.r6.txt');
tw_7 = csvread('./processed/rcv1.r7.txt');
tw_8 = csvread('./processed/rcv1.r8.txt');
tw_9 = csvread('./processed/rcv1.r9.txt');

twall = ( tw_0 + tw_1 + tw_2 + tw_3 + tw_4 + tw_5 + tw_6 + tw_7 + tw_8 + tw_9 ) / 10;


csvwrite('rcv1_twr.txt',twall);
