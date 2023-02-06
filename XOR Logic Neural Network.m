close all
clc
clear

%%%XOR nueron network

x1 = [0 0 1 1];
x2 = [0 1 0 1];
yt = [0 1 1 0];
w13 = 0.5;
w14 = 0.9;
w23 = 0.4;
w24 = 1;
w35 = -1.2;
w45 = 1.1;
th3 = 0.8;
th4 = -0.1;
th5 = 0.3;

sig = @(v) 1/(1+exp(-v));

e = 1;
a = 0.1;


while abs(e) >= 0.001
    for n = 1:4
        yh1 = sig(x1(n)*w13+x2(n)*w23-th3);
        yh2 = sig(x1(n)*w14+x2(n)*w24-th4);
        ya(n) = sig(yh1*w35+yh2*w45-th5);
        e = yt(n) - ya(n);
        
        s5 = ya(n)*(1-ya(n))*e;
        dw35 = a*yh1*s5;
        dw45 = a*yh2*s5;
        dth5 = a*(-1)*s5;
        
        s3 = yh1*(1-yh1)*s5*w35;
        dw13 = a*x1(n)*s3;
        dw23 = a*x2(n)*s3;
        dth3 = a*(-1)*s3;
        
        s4 = yh2*(1-yh2)*s5*w45;
        dw14 = a*x1(n)*s4;
        dw24 = a*x2(n)*s4;
        dth4 = a*(-1)*s4;
        
        w13 = w13 + dw13;
        w14 = w14 + dw14;
        w23 = w23 + dw23;
        w24 = w24 + dw24;
        w35 = w35 + dw35;
        w45 = w45 + dw45;
        
        th3 = th3 + dth3;
        th4 = th4 + dth4;
        th5 = th5 + dth5;
    end
    
    
end

%Check Network
x1 = [0 1 0 1]
x2 = [0 0 0 0]

for n = 1:4
yh1 = sig(x1(n)*w13+x2(n)*w23-th3);
yh2 = sig(x1(n)*w14+x2(n)*w24-th4);
ya(n) = sig(yh1*w35+yh2*w45-th5);
end
ya

%%

%%%Creat Network

x1 = [0 4 2 5];
x2 = [0 1 0 3];
yt = [1 0 1 0];
w11 = 3; w12 = -3; w13 = 3; w14 = -3; w15 = -1; w16 = 1; w17 = -1; w18 = 1;
w21 = -1; w22 = 1; w23 = 1; w24 = -1; w25 = -2; w26 = -2; w27 = 2; w28 = 2;
w19 = 1; w29 = 1; w39 = 1; w49 = 1; w59 = 1; w69 = 1; w79 = 1; w89 = 1;
th1 = -9; th2 = -9 ; th3 = -9; th4 = -9; th5 = -4; th6 = -4; th7 = -4; th8 = -4; th9 = 8;
sig = @(v) 1/(1+exp(-v));

e = 1;
a = 0.1;


while abs(e) >= 0.001
    for n = 1:4
        yh1 = sig(x1(n)*w11+x2(n)*w21-th1);
        yh2 = sig(x1(n)*w12+x2(n)*w22-th2);
        yh3 = sig(x1(n)*w13+x2(n)*w23-th3);
        yh4 = sig(x1(n)*w14+x2(n)*w24-th4);
        yh5 = sig(x1(n)*w15+x2(n)*w25-th5);
        yh6 = sig(x1(n)*w16+x2(n)*w26-th6);
        yh7 = sig(x1(n)*w17+x2(n)*w27-th7);
        yh8 = sig(x1(n)*w18+x2(n)*w28-th8);
        ya(n) = sig(yh1*w19+yh2*w29+yh3*w39+yh4*w49+yh5*w59+yh6*w69+yh7*w79+yh8*w89-th9);
        
        e = yt(n) - ya(n);
        
        s9 = ya(n)*(1-ya(n))*e;
        dw19 = a*yh1*s9;
        dw29 = a*yh2*s9;
        dw39 = a*yh3*s9;
        dw49 = a*yh4*s9;
        dw59 = a*yh5*s9;
        dw69 = a*yh6*s9;
        dw79 = a*yh7*s9;
        dw89 = a*yh8*s9;
        dth9 = a*(-1)*s9;
        
        s8 = yh8*(1-yh8)*s9*w89;
        dw18 = a*x1(n)*s8;
        dw28 = a*x2(n)*s8;
        dth8 = a*(-1)*s8;
        
        s7 = yh7*(1-yh7)*s9*w79;
        dw17 = a*x1(n)*s7;
        dw27 = a*x2(n)*s7;
        dth7 = a*(-1)*s7;
        
        s6 = yh6*(1-yh6)*s9*w69;
        dw16 = a*x1(n)*s6;
        dw26 = a*x2(n)*s6;
        dth6 = a*(-1)*s6;
        
        s5 = yh5*(1-yh5)*s9*w59;
        dw15 = a*x1(n)*s5;
        dw25 = a*x2(n)*s5;
        dth5 = a*(-1)*s5;
        
        s4 = yh4*(1-yh4)*s9*w49;
        dw14 = a*x1(n)*s4;
        dw24 = a*x2(n)*s4;
        dth4 = a*(-1)*s4;
        
        s3 = yh3*(1-yh3)*s9*w39;
        dw13 = a*x1(n)*s3;
        dw23 = a*x2(n)*s3;
        dth3 = a*(-1)*s3;
        
        s2 = yh2*(1-yh2)*s9*w29;
        dw12 = a*x1(n)*s2;
        dw22 = a*x2(n)*s2;
        dth2 = a*(-1)*s2;
        
        s1 = yh1*(1-yh1)*s9*w19;
        dw11 = a*x1(n)*s1;
        dw21 = a*x2(n)*s1;
        dth1 = a*(-1)*s1;
        
        w11 = w11 + dw11;
        w12 = w12 + dw12;
        w13 = w13 + dw13;
        w14 = w14 + dw14;
        w15 = w15 + dw15;
        w16 = w16 + dw16;
        w17 = w17 + dw17;
        w18 = w18 + dw18;
        w21 = w21 + dw21;
        w22 = w22 + dw22;
        w23 = w23 + dw23;
        w24 = w24 + dw24;
        w25 = w25 + dw25;
        w26 = w26 + dw26;
        w27 = w27 + dw27;
        w28 = w28 + dw28;
        w19 = w19 + dw19;
        w29 = w29 + dw29;
        w39 = w39 + dw39;
        w49 = w49 + dw49;
        w59 = w59 + dw59;
        w69 = w69 + dw69;
       w79 = w79 + dw79;
       w89 = w89 + dw89;
        
        th1 = th1 + dth1;
        th2 = th2 + dth2;
        th3 = th3 + dth3;
        th4 = th4 + dth4;
        th5 = th5 + dth5;
        th6 = th6 + dth6;
        th7 = th7 + dth7;
        th8 = th8 + dth8;
        th9 = th9 + dth9;
    end
    
    
end

%%
clc
x1 = [0 2 0 10]
x2 = [0 0 3 5]

for n = 1:4
            yh1 = sig(x1(n)*w11+x2(n)*w21-th1);
        yh2 = sig(x1(n)*w12+x2(n)*w22-th2);
        yh3 = sig(x1(n)*w13+x2(n)*w23-th3);
        yh4 = sig(x1(n)*w14+x2(n)*w24-th4);
        yh5 = sig(x1(n)*w15+x2(n)*w25-th5);
        yh6 = sig(x1(n)*w16+x2(n)*w26-th6);
        yh7 = sig(x1(n)*w17+x2(n)*w27-th7);
        yh8 = sig(x1(n)*w18+x2(n)*w28-th8);
        ya(n) = sig(yh1*w19+yh2*w29+yh3*w39+yh4*w49+yh5*w59+yh6*w69+yh7*w79+yh8*w89-th9);
end
ya