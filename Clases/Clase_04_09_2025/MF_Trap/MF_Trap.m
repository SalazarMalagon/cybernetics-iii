function [mu] = MF_Trap(x,a,b,c,d)
if x <= a
    mu = 0;
elseif x>a && x<b 
    mu = (x-a)/(b-a);
elseif x>=b && x<=c
    mu=1;
elseif x>c && x<d
    mu=(d-x)/(d-c);
else
    mu=0;
end
end
