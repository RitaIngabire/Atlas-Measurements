function PoA_student_workshop(d, mu)


d=[30 10 20 50 40]; %ms
mu=[15 35 25 55 40]./1000; %services per ms

%compute results for M/M/1, yes/not
mm1 = 1;

%compute results for M/D/1
md1 = 1;

%number of points per curve
nruns = 201;

ignore_d = 0;

lambda_set = linspace(0,0.999*sum(mu),nruns)

dmin = d + 1./mu;
[dmin_sorted order] = sort(dmin - ignore_d * d);
d_sorted = d(order);
d_sorted_used = d(order) * (1-ignore_d);
mu_sorted = mu(order);

for j = 2:numel(mu)
    if mm1
        lambda_th_nep_mm1(j) = sum(mu_sorted(1:j-1)  - 1./(d_sorted_used(j)-d_sorted_used(1:j-1)+1/mu_sorted(j)));
        lambda_th_opt_mm1(j) = sum(mu_sorted(1:j-1)  - sqrt((mu_sorted(1:j-1))./(d_sorted_used(j)-d_sorted_used(1:j-1)+1/mu_sorted(j))));
    end
    if md1
        lambda_th_nep_md1(j) = 2 * sum(mu_sorted(1:j-1)  .* (mu_sorted(1:j-1) .* (d_sorted_used(j)-d_sorted_used(1:j-1)+1/mu_sorted(j))-1)./(2 * mu_sorted(1:j-1) .* (d_sorted_used(j)-d_sorted_used(1:j-1)+1/mu_sorted(j))-1));
        lambda_th_opt_md1(j) = sum(mu_sorted(1:j-1)  .* ( 1 - 1./sqrt(2*mu_sorted(1:j-1).*(d_sorted_used(j)-d_sorted_used(1:j-1)+1/mu_sorted(j)) - 1) ) );
    end
    
end

if mm1
    res_mm1 = zeros (nruns + 2*(numel(mu)-1),2+2*numel(mu));
    lambda_set = [lambda_set lambda_th_nep_mm1 lambda_th_opt_mm1]; 
end
if md1
    res_md1 = zeros (nruns + 2*(numel(mu)-1),2+2*numel(mu));
    lambda_set = [lambda_set lambda_th_nep_md1 lambda_th_opt_md1]; 
end

lambda_set = sort(lambda_set); 

cnt=0;

for lambda = lambda_set
    cnt = cnt + 1;
    fprintf('\\rho = %g\n',lambda/sum(mu))

    if mm1
        if lambda >= lambda_th_opt_mm1(end)
            jstar_mm1 = numel(lambda_th_opt_mm1);
        else
            jstar_mm1 = 1;
            while lambda >= lambda_th_opt_mm1(jstar_mm1)
                jstar_mm1 = jstar_mm1 + 1;
            end
            jstar_mm1 = jstar_mm1 - 1;
        end

        if lambda >= lambda_th_nep_mm1(end)
            jdagger_mm1 = numel(lambda_th_nep_mm1);
        else
            jdagger_mm1 = 1;
            while lambda >= lambda_th_nep_mm1(jdagger_mm1)
                jdagger_mm1 = jdagger_mm1 + 1;
            end
            jdagger_mm1 = jdagger_mm1 - 1;
        end
        res_mm1(cnt,:) = poa_multiple_mm1(d_sorted, mu_sorted, order, lambda, jstar_mm1, jdagger_mm1, ignore_d);
    end

    if md1
        if lambda >= lambda_th_opt_md1(end)
            jstar_md1 = numel(lambda_th_opt_md1);
        else
            jstar_md1 = 1;
            while lambda >= lambda_th_opt_md1(jstar_md1)
                jstar_md1 = jstar_md1 + 1;
            end
            jstar_md1 = jstar_md1 - 1;
        end

        if lambda >= lambda_th_nep_md1(end)
            jdagger_md1 = numel(lambda_th_nep_md1);
        else
            jdagger_md1 = 1;
            while lambda >= lambda_th_nep_md1(jdagger_md1)
                jdagger_md1 = jdagger_md1 + 1;
            end
            jdagger_md1 = jdagger_md1 - 1;
        end
        res_md1(cnt,:) = poa_multiple_md1(d_sorted, mu_sorted, order, lambda, jstar_md1, jdagger_md1, ignore_d);
    end
end


%M/M/1
if mm1
    figure('Position',[10 10 900 600]);
    pl_opt=plot(lambda_set/sum(mu),res_mm1(:,2),'LineWidth',2,'DisplayName','Opt');
    hold on
    pl_nep=plot(lambda_set/sum(mu),res_mm1(:,1),'LineWidth',2,'DisplayName','NEP');
    for j = 2:numel(mu)
        xline(lambda_th_nep_mm1(j)/sum(mu), '-.', 'Color', pl_opt.Color, 'DisplayName',sprintf('Server %d on (NEP)',j),'HandleVisibility','off');
        xline(lambda_th_opt_mm1(j)/sum(mu), '-.', 'Color', pl_nep.Color,  'DisplayName',sprintf('Server %d on (Opt)',j),'HandleVisibility','off');
    end
    %lambda_subset = lambda_set( ceil(numel(lambda_set)/2) : end);
    %plot(lambda_subset/sum(mu),((sum(sqrt(mu)))^2)./(sum(mu) - lambda_subset)./lambda_subset,':', 'Color', pl_opt.Color,'LineWidth',2,'DisplayName','Opt (approx)')
    %plot(lambda_subset/sum(mu),numel(d)./(sum(mu) - lambda_subset),':', 'Color', pl_nep.Color,'LineWidth',2,'DisplayName','NEP (approx)')

    %plot(lambda_set/sum(mu),bound,'LineWidth',2,'DisplayName','LB for Opt')
    set(gca, 'YScale', 'log')
    legend('boxoff')
    legend('location','best')
    xlabel('\rho')
    ylabel('Average system latency (ms)')
    set(gca,'fontsize',18)
    title('Round-trip average latency with M/M/1 servers')

    figure('Position',[300 10 900 600]);
    pl_poa=plot(lambda_set/sum(mu),res_mm1(:,1)./res_mm1(:,2),'k','LineWidth',2,'DisplayName','PoA');
    hold on
    for j = 2:numel(mu)
        xline(lambda_th_nep_mm1(j)/sum(mu), '-.', 'Color', pl_opt.Color, 'DisplayName',sprintf('Server %d on (NEP)',j),'HandleVisibility','off');
        xline(lambda_th_opt_mm1(j)/sum(mu), '-.', 'Color', pl_nep.Color, 'DisplayName',sprintf('Server %d on (OPT)',j),'HandleVisibility','off');
    end
    plot(1,numel(d) * sum(mu) ./ (sum(sqrt(mu)))^2 ,'p', 'Color', pl_poa.Color, 'markersize',10,'LineWidth',2,'DisplayName','PoA at full load')
    if ignore_d==0
        yup = ylim;
        ylim([1 yup(2)])
    end
    legend('boxoff')
    legend('location','best')
    xlabel('\rho')
    ylabel('Price of anaychy (\eta)')
    set(gca,'fontsize',18)
    title('Price of anarchy with M/M/1 servers')
end

%%%%% MD1
if md1
    figure('Position',[600 10 900 600]);
    pl_opt=plot(lambda_set/sum(mu),res_md1(:,2),'LineWidth',2,'DisplayName','Opt');
    hold on
    pl_nep=plot(lambda_set/sum(mu),res_md1(:,1),'LineWidth',2,'DisplayName','NEP');
    for j = 2:numel(mu)
        xline(lambda_th_nep_md1(j)/sum(mu), '-.', 'Color', pl_opt.Color, 'DisplayName',sprintf('Server %d on (NEP)',j),'HandleVisibility','off');
        xline(lambda_th_opt_md1(j)/sum(mu), '-.', 'Color', pl_nep.Color, 'DisplayName',sprintf('Server %d on (Opt)',j),'HandleVisibility','off');
    end
    %lambda_subset = lambda_set( ceil(numel(lambda_set)/2) : end);
    %plot(lambda_subset/sum(mu),0.5 * ((sum(sqrt(mu)))^2)./(sum(mu) - lambda_subset)./lambda_subset,':', 'Color', pl_opt.Color,'LineWidth',2,'DisplayName','Opt (approx)')
    %plot(lambda_subset/sum(mu),0.5 * numel(d)./(sum(mu) - lambda_subset),':', 'Color', pl_nep.Color,'LineWidth',2,'DisplayName','NEP (approx)')

    set(gca, 'YScale', 'log')
    legend('boxoff')
    legend('location','best')
    xlabel('\rho')
    ylabel('Average system latency (ms)')
    set(gca,'fontsize',18)
    title('Round-trip average latency with M/D/1 servers')


    figure('Position',[900 10 900 600]);
    pl_poa=plot(lambda_set/sum(mu),res_md1(:,1)./res_md1(:,2),'k','LineWidth',2,'DisplayName','PoA');
    %set(gca, 'YScale', 'log')
    hold on
    for j = 2:numel(mu)
        xline(lambda_th_nep_md1(j)/sum(mu), '-.', 'Color', pl_opt.Color, 'DisplayName',sprintf('Server %d on (NEP)',j),'HandleVisibility','off');
        xline(lambda_th_opt_md1(j)/sum(mu), '-.', 'Color', pl_nep.Color, 'DisplayName',sprintf('Server %d on (OPT)',j),'HandleVisibility','off');
    end
    plot(1,numel(d) * sum(mu) ./ (sum(sqrt(mu)))^2 ,'p', 'Color', pl_poa.Color, 'markersize',10,'LineWidth',2,'DisplayName','PoA at full load')
    if ignore_d==0
        yup = ylim;
        ylim([1 yup(2)])
    end
    legend('boxoff')
    legend('location','best')
    xlabel('\rho')
    ylabel('Price of anaychy (\eta)')
    set(gca,'fontsize',18)
    title('Price of anarchy with M/D/1 servers')

end
end



function res = poa_multiple_mm1(d, mu, order, lambda, jstar, jdagger, ignore_d)

%KKT-based optimization

gamma = vmbinarysolve_opt(d(1:jstar) * (1-ignore_d), mu(1:jstar),lambda,Inf);
pi_kkt(1:jstar) = (mu(1:jstar) - sqrt(mu(1:jstar) ./ (gamma - d(1:jstar)* (1-ignore_d))) ) / lambda;
if jstar < numel(d)
    pi_kkt(jstar+1:numel(d)) = 0; 
end
w_kkt = sum(pi_kkt .* (d + 1 ./ (mu - pi_kkt*lambda)) );


%NEP

alpha = vmbinarysolve_nep(d(1:jdagger) * (1-ignore_d), mu(1:jdagger),lambda,Inf);
pi_nep(1:jdagger) = (mu(1:jdagger) - 1 ./ (alpha - d(1:jdagger)* (1-ignore_d))) / lambda;
if jdagger < numel(d)
    pi_nep(jdagger+1:numel(d)) = 0; 
end
w_nep = sum(pi_nep .* (d + 1 ./ (mu - pi_nep*lambda)) ); %this should be equal to alpha


%==========

fprintf('M/M/1: NE with %d ser.',jdagger)
fprintf(': %g ms at p = [ ', w_nep)
for j = 1 : numel(d)
    pos = find(order==j);
    fprintf('%g ',pi_nep(pos))
end
fprintf(']');

fprintf('| KKT latency: %g ms at p = [ ', w_kkt)
for j = 1 : numel(d)
    pos = find(order==j);
    fprintf('%g ',pi_kkt(pos))
    res(2+numel(mu)+j) = pi_kkt(pos);
end
fprintf(']');

fprintf('| PoA: %g', w_nep / w_kkt)
fprintf(' | gamma = %g, alpha = %g', gamma, alpha)
fprintf('\n')

res(1:2) = [w_nep w_kkt];

end


function g = vmbinarysolve_opt(d,mu,lambda,gmax)
gmin = d(end)+1/mu(end);
if isinf(gmax)
    gmax = gmin*10;
    while sum(mu - sqrt(mu ./ (gmax - d))) - lambda < 0
        gmin = gmax;
        gmax = gmax*10;
    end
end
while gmax - gmin > 1.0e-9
    g = 0.5 * (gmin + gmax);
    val = sum(mu - sqrt(mu ./ (g - d))) - lambda;
    if val > 0
        gmax = g;
    elseif val < 0
        gmin = g;
    else 
        gmin = g;
        gmax = g; 
    end
end
g = 0.5 * (gmin + gmax);
end

function g = vmbinarysolve_nep(d,mu,lambda,gmax)
gmin = d(end)+1/mu(end);
if isinf(gmax)
    gmax = gmin*10;
    while sum(mu - 1 ./ (gmax - d)) - lambda < 0
        gmin = gmax;
        gmax = gmax*10;
    end
end
while gmax - gmin > 1.0e-9
    g = 0.5 * (gmin + gmax);
    val = sum(mu - 1 ./ (g - d)) - lambda;
    if val > 0
        gmax = g;
    elseif val < 0
        gmin = g;
    else 
        gmin = g;
        gmax = g; 
    end
end
g = 0.5 * (gmin + gmax);
end

function res = poa_multiple_md1(d, mu, order, lambda, jstar, jdagger, ignore_d)

%KKT-based optimization

gamma = vmbinarysolve_md1_opt(d(1:jstar) * (1-ignore_d), mu(1:jstar),lambda,Inf);
pi_kkt(1:jstar) = mu(1:jstar) .* (1 - 1 ./ sqrt( 2 * mu(1:jstar) .* (gamma - d(1:jstar)* (1-ignore_d)) - 1 ) ) / lambda;
if jstar < numel(d)
    pi_kkt(jstar+1:numel(d)) = 0; 
end
w_kkt = sum(pi_kkt .* (d + 1 ./ (2 * mu) .* (2 - pi_kkt*lambda./mu) ./ (1 - pi_kkt*lambda./mu)));


%NEP

alpha = vmbinarysolve_md1_nep(d(1:jdagger) * (1-ignore_d), mu(1:jdagger),lambda,Inf);
pi_nep(1:jdagger) = 2 * mu(1:jdagger) .* ( mu(1:jdagger) .* ( alpha - d(1:jdagger) * (1-ignore_d) ) - 1 ) ./ ( 2 * mu(1:jdagger) .* ( alpha - d(1:jdagger) * (1-ignore_d) ) - 1 ) / lambda;
if jdagger < numel(d)
    pi_nep(jdagger+1:numel(d)) = 0; 
end
w_nep = sum(pi_nep .* (d + 1 ./ (2 * mu) .* (2 - pi_nep*lambda./mu) ./ (1 - pi_nep*lambda./mu))); %this should be equal to alpha


%==========

fprintf('M/D/1: NE with %d ser.',jdagger)
fprintf(': %g ms at p = [ ', w_nep)
for j = 1 : numel(d)
    pos = find(order==j);
    fprintf('%g ',pi_nep(pos))
end
fprintf(']');

fprintf('| KKT latency: %g ms at p = [ ', w_kkt)
for j = 1 : numel(d)
    pos = find(order==j);
    fprintf('%g ',pi_kkt(pos))
    res(2+numel(mu)+j) = pi_kkt(pos);
end
fprintf(']');

fprintf('| PoA: %g', w_nep / w_kkt)
fprintf(' | gamma = %g, alpha = %g', gamma, alpha)
fprintf('\n')

res(1:2) = [w_nep w_kkt];

end


function g = vmbinarysolve_md1_opt(d,mu,lambda,gmax)
gmin = d(end)+1/mu(end);
if isinf(gmax)
    gmax = gmin*10;
    while sum(mu .* (1 - 1 ./ sqrt( 2 * mu .* (gmax - d) - 1 ) )) - lambda < 0
        gmin = gmax;
        gmax = gmax*10;
    end
end
while gmax - gmin > 1.0e-9
    g = 0.5 * (gmin + gmax);
    val = sum(mu .* (1 - 1 ./ sqrt( 2 * mu .* (g - d) - 1 ) )) - lambda;
    if val > 0
        gmax = g;
    elseif val < 0
        gmin = g;
    else 
        gmin = g;
        gmax = g; 
    end
end
g = 0.5 * (gmin + gmax);
end

function g = vmbinarysolve_md1_nep(d,mu,lambda,gmax)
gmin = d(end)+1/mu(end);
if isinf(gmax)
    gmax = gmin*10;
    while sum(2 * mu .* ( mu .* ( gmax - d ) - 1 ) ./ ( 2 * mu .* ( gmax - d ) - 1 )) - lambda < 0
        gmin = gmax;
        gmax = gmax*10;
    end
end
while gmax - gmin > 1.0e-9
    g = 0.5 * (gmin + gmax);
    val = sum(2 * mu .* ( mu .* ( g - d ) - 1 ) ./ ( 2 * mu .* ( g - d ) - 1 )) - lambda;
    if val > 0
        gmax = g;
    elseif val < 0
        gmin = g;
    else 
        gmin = g;
        gmax = g; 
    end
end
g = 0.5 * (gmin + gmax);
end




