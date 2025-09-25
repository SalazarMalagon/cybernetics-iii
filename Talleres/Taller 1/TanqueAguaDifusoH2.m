function ft = TanqueAguaDifusoH2(h)

B = smf(h, [0.045 0.055]);   % sensor B pasa a 1 cerca de 0.05
M = smf(h, [0.55 1.12]);   % sensor M pasa a 1 cerca de 1.1
C = smf(h, [1.9  1.97]);   % sensor C pasa a 1 cerca de 1.95
A = smf(h, [2 2.1]);   % sensor A pasa a 1 cerca de 2.00

Y1 = 1 - A;
Y2 = max( 1 - M, min(1 - A, C) );
Y3 = 1 - C;

ft = 0.07*Y1 + 0.11*Y2 + 0.15*Y3;
end