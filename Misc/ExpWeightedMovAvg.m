

close all
clear

beta=0.5;

load motorcycledata
plot(times,gmeasurements)
hold on

N=length(gmeasurements);

smoothed=0;
for i=1:N
    smoothed=beta*smoothed+(1-beta)*gmeasurements(i);
    plot(times(i),smoothed,'r.','MarkerSize',15);
end
grid
