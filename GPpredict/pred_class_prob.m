function [b, lpG ] = pred_class_prob( x, y, tG, HyperParams)

disp('Using Correct Class')
%%% This is the classification routine %%%%
SPARSEFLAG=0;
    dim=size(tG,2);

        Ind=HyperParams.NumInducingClass;
        sparse=nthroot(Ind,dim);
    meanfunc = @meanConst; 
    covfunc = @covSEard; %%% I have found 300 inducing points to be enough for this problem

    likfunc = @likErf;
    if dim==4
    [u1,u2,u3,u4] = ndgrid(linspace(0,1,sparse)); 
    u = [u1(:),u2(:),u3(:), u4(:)]; 
    clear u1; clear u2;clear u3;
    elseif dim==3
    [u1,u2,u3] = ndgrid(linspace(0,1,sparse)); 
    u = [u1(:),u2(:),u3(:)]; 
    clear u1; clear u2;clear u3;
    else
    [u1,u2] = ndgrid(linspace(0,1,sparse)); 
    u = [u1(:),u2(:)]; 
    clear u1; clear u2;
    end
    nu = size(u,1);
    covfuncF = {@covFITC, {covfunc},u};
    inffunc = @infFITC_EP;%@infMCMC;%  
    % also @infFITC_Laplace is possible
    hyp.cov = log(ones(dim+1,1)); 
    hyp.mean=0;
    hyp.xu=u;
    n=length(x);
    nt=length(tG);
    if n > HyperParams.sparseMargin || SPARSEFLAG==1
       disp('SPARSE')
    if HyperParams.minimize==1
    hyp = minimize(hyp, @gp, -300, inffunc, meanfunc, covfuncF, likfunc,x, y); %uncomment when covariance parameter is to be learnt
    end
    [a b c d lpG] = gp(HyperParams.hyp, inffunc, meanfunc, covfuncF, likfunc, x, y,tG, ones(nt, 1));
    else
        disp('FULL')
    if HyperParams.minimize==1
    hyp = minimize(hyp, @gp, -300, @infEP, meanfunc, covfunc, likfunc, x, y);
    end
    [a b c d lpG] = gp(HyperParams.hyp, @infEP, meanfunc, covfunc, likfunc, x, y,tG, ones(nt, 1));
    end
end

