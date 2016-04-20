function [ layer,forward_handle,backward_handle ] = createUncertainLayer( input_args )
%function [ layer,forward_handle,backward_handle ] = createUncertainLayer( input_args )
% create a layer with forward and backward handle. the layer calculates the
% amount of certainty with different methods (entropy implemented)
% the layer has a handle forward
% res(end -1).x is a 1*1*C*B array. 
% res(end).x should be B
%
%
% forward_handle()
% function resip1 = calcUncertainty(layer,resi,resip1)
%
%
% backward
% function resi = calcUncertaintyder(layer,resi,resip1)

layer = struct('type','custom' , ...
'unctype', 'entropy',...
'forward', @calcUncertainty,...
'backward', @calcUncertaintyder);
forward_handle = @calcUncertainty;
backward_handle = @calcUncertaintyder;






end


function out = calcUncertainty(layer,resi,resip1)
%% assumes that in is not softmaxed
% function out = calcUncertainty(layer,in,dzdy)
% if the function has 2 inputs it will produce the forward pass depending
% on the layer type.
    if  strcmp (layer.unctype, 'entropy')
            out = resip1;
            softIn = calcSoftmax(layer,resi.x);
            out.x = 1 - calcEntropySoft(layer,softIn);
        

    else 
        assert(false,'uncertainty type unknown');
    end
end
function out = calcUncertaintyder(layer,resi,resip1)
    if strcmp(layer.unctype,'entropy')
            out = resi;
            p = calcSoftmax(layer,resi.x);
            dzdp = calcEntropySoft(layer,p,-resip1.dzdx);
            
            out.dzdx = calcSoftmax(layer,resi.x,dzdp);
    else
        assert(false,'uncertainty type unknown')
    end
end
function out = calcSoftmax(layer,in,dzdy)
    %% normalizes the input by softmax function
    % if number of inputs is 2 forward passes
    % input is of size 1*1*C*B
    % out is of size 1*1*C*B
    % if number of the inputs is 3 backward pass and outputs the dzdx
    % in case of derrivative out is 1*1*C*B
    classNum = size(in,3);
    if nargin <= 2
        assert(nargin == 2,'invalid number of inputs');
      % gets the sigmoid
        MAX = max(in,[],3);
        sigedIn = bsxfun(@minus,in,MAX);
        sigedIn =  exp(sigedIn);
      % normalize the sigmoid;  sigedIn is 1*1*C*B
        out = bsxfun(@times,sigedIn, 1./sum(sigedIn,3));
        
        
    else
           batchNum = size(in,4);
           out  = sigmoid(in,'sigmoid');
           
        % normalize the sigmoid;  sigedIn is 1*1*C*B
           % scalex.. shows 1-sigm(xj) at 1,~,j,B
           
           Px_x_j_B = repmat(out,1,classNum,1,1); 
           Px_j_x_B = permute(out,[1,3,2,4]);
           Px_j_x_B = repmat(Px_j_x_B,1,1,classNum,1);
           % finding the eye indices of 2nd and 3rd dim
           fourthindex = repmat(1:batchNum,classNum,1);
           fourthindex = fourthindex(:);
           eyeIndices = sub2ind(size(Px_j_x_B),ones(size(fourthindex)),repmat(1:classNum,1,batchNum)',repmat(1:classNum,1,batchNum)',fourthindex);
           % building the output
           
           dpidxjx_i_j_B = zeros(size(Px_j_x_B),'gpuArray');
           dpidxjx_i_j_B(eyeIndices) = Px_j_x_B(eyeIndices);
           dpidxjx_i_j_B =(dpidxjx_i_j_B  -Px_j_x_B .* Px_x_j_B) ;
           % calc dzdx  dpidxjx_i... is 1*c*c*b
           dzdpix_i_x_B = permute(dzdy,[1,3,2,4]);
           % dzdpix_i_x_B is shows piB
           dzdpix_i_x_B = repmat(dzdpix_i_x_B,1,1,classNum,1);
           % dzdpix_i_x_B is now 1*c*C*B dims
           out = dzdpix_i_x_B .* dpidxjx_i_j_B;
           out = sum(out,2);
           
      
           
    end
end

function out =  calcEntropySoft(layer,In,dzdy)
    %% calculates entropy with softmaxed inputs(Probabilities);
    % In is a 1*1*C*B dimensions
    % out is a 1*1*1*B
    if nargin <= 2
        In = -In .* log( In)/log(size(In,3));
        In(isnan(In)) = 0;
        % calculate summation of the Information ; sigedIn is 1*1*C*B
        out = sum(In,3);
    else
        classNum = size(In,3);
        
        out = (-(log(In)/log(classNum))- (1/log(classNum)))*dzdy;
        out(isinf(out)) = 0;
        out = bsxfun(@minus,out,sum(out,3));
    end
end
