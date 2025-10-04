function ft = TanqueAguaDifuso(h)

B = smf(h, [0.045 0.055]);   % sensor B pasa a 1 cerca de 0.05
M = smf(h, [0.53  0.66 ]);   % sensor M pasa a 1 cerca de 0.60
C = smf(h, [0.92  0.98 ]);   % sensor C pasa a 1 cerca de 0.95
A = smf(h, [0.975 1.03]);   % sensor A pasa a 1 cerca de 1.00

Y1 = 1 - A;
Y2 = max( 1 - M, min(1 - A, C) );
Y3 = 1 - C;

ft = 0.02*Y1 + 0.04*Y2 + 0.06*Y3;
end