function [Choice]= simple_model2(start_point, sigma, bias, dt, time_interval)
p = normcdf(start_point,bias*dt,sigma*sqrt(dt));
x = rand;
Choice = x > p;
end