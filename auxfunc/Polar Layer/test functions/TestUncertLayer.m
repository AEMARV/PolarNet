function out = TestUncertLayer()
x = gpuArray(single(ones(1,1,4,4)));
x(1,1,:,1) = [0,100,0,0];
x(1,1,:,2) = [5,10,5,0];
x(1,1,:,3) = [-10,100,1,4];
x(1,1,:,4) = [-10,-10,-10,-10];
resi = struct('x',x,'dzdx',[]);
resip1 = struct('x',[],'dzdx',1);
[layer,forwardHandle,backwardHandle] = createUncertainLayer();
resip1 = forwardHandle(layer,resi,resip1);
fx = resip1.x;
resip = backwardHandle(layer,resi,resip1);
dfxanal = resip.dzdx;
before = resip1.x;
%% emperical
epsilon = 0.001;
deriv = x;
x1Emp = x;
for i = 1:4
    x1Emp = x;
x1Emp(:,:,i,:) = x1Emp(:,:,i,:) + epsilon;
empResi = resip1;
empResip1 = resip1;
empResi.x = x1Emp;
empResip1 = forwardHandle(layer,empResi,empResip1);
fx2 = empResip1.x;
empResi.x = resip1.x;
deriv(:,:,i,:) = (empResip1.x(:) - empResi.x(:))/epsilon;
end



deriv == resip.dzdx;


resip.x = resip.x + deriv;
resip1 = forwardHandle(layer,resip,resip1);
after = resip1.x;

difff = after -before