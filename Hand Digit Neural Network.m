close all
clc
clear

%%%loading data

data = load('data');
datay = data.y;
datax = data.X;

loca = randi([1,5000],100,1);
for i = 1:length(loca)
    M = datax(loca(i,1),:);
    SelsectData{i} = reshape(M,20,[]);
end

n = 1;
for i = 1:10:100
    I{n} = cat(2,SelsectData{i},SelsectData{i+1:i+9});
    n = n+1;
end
image = cat(1,I{1},I{2});
for i = 3:10
    image = cat(1,image,I{i});
end

imshow(image)
%%
%%%train and test data
clc
train = zeros(3000,400);

for i = 1:10
    s1 = 500*(i-1)+1;
    s2 = s1 + 299;
    tr{i} = datax(s1:s2,:);
    ts{i} = datax(s2+1:s2+200,:);
    yr{i} = datay(s1:s2,:);
    ys{i} = datay(s2+1:s2+200,:);
end

train = cat(1,tr{1},tr{2});
for i = 3:10
    train = cat(1,train,tr{i});
end

test = cat(1,ts{1},ts{2});
for i = 3:10
    test = cat(1,test,ts{i});
end

y_test = cat(1,ys{1},ys{2});
for i = 3:10
    y_test = cat(1,y_test,ys{i});
end

y_train = cat(1,yr{1},yr{2});
for i = 3:10
    y_train = cat(1,y_train,yr{i});
end

%%
%%%creat neuron network

Y = zeros(3000,10);

for i = 1:length(y_train)
    switch y_train(i)
        case 10
            Y(i,:) = [1,0,0,0,0,0,0,0,0,0] ;
        case 1
            Y(i,:) = [0,1,0,0,0,0,0,0,0,0] ;
        case 2
            Y(i,:) = [0,0,1,0,0,0,0,0,0,0] ;
        case 3
            Y(i,:) = [0,0,0,1,0,0,0,0,0,0] ;
        case 4
            Y(i,:) = [0,0,0,0,1,0,0,0,0,0] ;
        case 5
            Y(i,:) = [0,0,0,0,0,1,0,0,0,0] ;
        case 6
            Y(i,:) = [0,0,0,0,0,0,1,0,0,0] ;
        case 7
            Y(i,:) = [0,0,0,0,0,0,0,1,0,0] ;
        case 8
            Y(i,:) = [0,0,0,0,0,0,0,0,1,0] ;
        case 9
            Y(i,:) = [0,0,0,0,0,0,0,0,0,1] ;
    end
end

%%
clc
%%%BackPropagation

input_layer_size = 400; hidden_layer_size = 25; num_labels = 10;
eps = 0.12;

W12 = rand(hidden_layer_size,1+input_layer_size)*2*eps-eps;
W23 = rand(num_labels,1+hidden_layer_size)*2*eps-eps;

initial_Theta1 = W12;
initial_Theta2 = W23;

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
X = train;
lambda = 1;

[J, grad] = nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y, lambda);

%%
%%%Bulif Neural Network
clc
costFunction = @(p) nnCostFunction(p, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, X, Y, lambda);

options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

%%
%%%Output Layer Neural Network
m = 2000;
r = randperm(m);

for i = 1:25
    subplot(5,5,i);
    I = reshape(test(r(i),:),20,[]);
    imshow(I)
    pred = findlabel(Theta1, Theta2, test(r(i),:));
    xlabel(pred-1)
end
%%
%%%Hidden Layer Neural Network
m = 2000;
r = randperm(m);

for i = 1:25
    subplot(5,5,i);
    x = test(r(i),:);
    [p,h1] = findhiddenlayer(Theta1 , x);
    I = reshape(h1,5,[]);
    imshow(I)
end

%%
%%%Calculate Accurancy
clc
error = zeros(2000,1);

for i = 1:2000
    pred = findlabel(Theta1, Theta2, test(i,:));
    if (pred-1) == mod(y_test(i),10)
        error(i,1) = 1;
    end
end
accurancy = nnz(error)/2000*100

%%
%%%Functions


function y = sigmoid(z)
y = 1./(1+exp(-z));
end

function y = diffsigmoid(z)
y = sigmoid(z).*(1-sigmoid(z));
end

function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function. Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives. The Polack-
% Ribiere flavour of conjugate gradients is used to compute search directions,
% and a line search using quadratic and cubic polynomial approximations and the
% Wolfe-Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes. Additionally a bunch of checks are made to
% make sure that exploration is taking place and that extrapolation will not
% be unboundedly large. The "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The function returns when either its length is up, or if no further
% progress can be made (ie, we are at a minimum, or so close that due to
% numerical problems, we cannot get any closer). If the function terminates
% within a few iterations, it could be an indication that the function value
% and derivatives are not consistent (ie, there may be a bug in the
% implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%
% See also: checkgrad
%
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
%
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
%
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%
% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
    argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
    i = i + (length>0);                                      % count iterations?!
    
    X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
    X = X + z1*s;                                             % begin line search
    [f2 df2] = eval(argstr);
    i = i + (length<0);                                          % count epochs?!
    d2 = df2'*s;
    f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
    if length>0, M = MAX; else M = min(MAX, -length-i); end
    success = 0; limit = -1;                     % initialize quanteties
    while 1
        while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0)
            limit = z1;                                         % tighten the bracket
            if f2 > f1
                z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
            else
                A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
                B = 3*(f3-f2)-z3*(d3+2*d2);
                z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
            end
            if isnan(z2) || isinf(z2)
                z2 = z3/2;                  % if we had a numerical problem then bisect
            end
            z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
            z1 = z1 + z2;                                           % update the step
            X = X + z2*s;
            [f2 df2] = eval(argstr);
            M = M - 1; i = i + (length<0);                           % count epochs?!
            d2 = df2'*s;
            z3 = z3-z2;                    % z3 is now relative to the location of z2
        end
        if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
            break;                                                % this is a failure
        elseif d2 > SIG*d1
            success = 1; break;                                             % success
        elseif M == 0
            break;                                                          % failure
        end
        A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
        if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 % num prob or wrong sign?
            if limit < -0.5                               % if we have no upper limit
                z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
            else
                z2 = (limit-z1)/2;                                   % otherwise bisect
            end
        elseif (limit > -0.5) && (z2+z1 > limit)         % extraplation beyond max?
            z2 = (limit-z1)/2;                                               % bisect
        elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
            z2 = z1*(EXT-1.0);                           % set to extrapolation limit
        elseif z2 < -z3*INT
            z2 = -z3*INT;
        elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))  % too close to limit?
            z2 = (limit-z1)*(1.0-INT);
        end
        f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
        z1 = z1 + z2; X = X + z2*s;                      % update current estimates
        [f2 df2] = eval(argstr);
        M = M - 1; i = i + (length<0);                             % count epochs?!
        d2 = df2'*s;
    end                                                      % end of line search
    
    if success                                         % if line search succeeded
        f1 = f2; fX = [fX' f1]';
        fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
        s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
        tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
        d2 = df1'*s;
        if d2 > 0                                      % new slope must be negative
            s = -df1;                              % otherwise use steepest direction
            d2 = -s'*s;
        end
        z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
        d1 = d2;
        ls_failed = 0;                              % this line search did not fail
    else
        X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
        if ls_failed || i > abs(length)          % line search failed twice in a row
            break;                             % or we ran out of time, so we give up
        end
        tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
        s = -df1;                                                    % try steepest
        d1 = -s'*s;
        z1 = 1/(1-d1);
        ls_failed = 1;                                    % this line search failed
    end
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end
fprintf('\n');
end

function [J, grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y, lambda)


%W12 = rand(hidden_layer_size,1+input_layer_size)*2*eps-eps;
%W23 = rand(num_labels,1+hidden_layer_size)*2*eps-eps;

%teta1 = W12;
%teta2 = W23;
teta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

teta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

a1 = [ones(size(X,1),1) X];
z2 = a1*teta1';
a2 = [ones(size(z2,1),1) sigmoid(z2)];
z3 = a2*teta2';
a3 = sigmoid(z3);
h = a3;
m = size(X,1);


p = sum(sum(teta1(:, 2:end).^2, 2))+sum(sum(teta2(:, 2:end).^2, 2));
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);

% calculate sigmas
sigma3 = a3 -Y;
sigma2 = (sigma3*teta2).*diffsigmoid([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);


% accumulate gradients
delta_1 = (sigma2'*a1);
delta_2 = (sigma3'*a2);

% calculate regularized gradient
p1 = (lambda/m).*[zeros(size(teta1, 1), 1) teta1(:, 2:end)];
p2 = (lambda/m).*[zeros(size(teta2, 1), 1) teta2(:, 2:end)];

Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

function p = findlabel(Theta1, Theta2, x)
m = size(x, 1);
%p = zeros(size(x, 1), 1);
h1 = sigmoid([ones(m, 1) x] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[d, p] = max(h2, [], 2);
end

function [p,h1] = findhiddenlayer(Theta1 , x)
m = size(x, 1);
%p = zeros(size(x, 1), 1);
h1 = sigmoid([ones(m, 1) x] * Theta1');
[d, p] = max(h1, [], 2);
end