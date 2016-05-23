
function [output_args_dzdx,DataParamDer]  = pol_transform( input_args,DataParam,dzdout)
% function [output_args/dzdx,DataParamDer]  = pol_transform( input_args,DataParam,dzdout )
%
% input_args is a matrix of size H*W*C*B
%
%
% DataParam is a struct     [
%                           rmin : 0:inf
%                           rmax : 0:inf
%                           row0 : -1:1
%                           col0 : -1:1
%                           theta0 : 0:2pi or -inf:inf
%                           type : it is not used
%                          ]
%--------------------------------------------------------------------------
% DataParamDer is a struct [
%                           dzdrmin :
%                           dzdrmax :
%                           .
%                           .
%                           .
%                           dzdtheta0
%                           ]
doder = nargout >1;
if ~doder
    rmin = DataParam.rmin; % 0 : inf
    rmax = DataParam.rmax; % 0 : inf
    row0 = DataParam.row0;% -1 : 1
    col0 = DataParam.col0; % -1 : 1
    theta0 = DataParam.theta0; % 0 : 2pi
    SIZE = size(input_args,2);
    Grid = createGrid(rmin,rmax,theta0,row0,col0,SIZE);
    output_args_dzdx = vl_nnbilinearsampler(input_args,Grid);
else
    rmin = DataParam.rmin;
    rmax = DataParam.rmax;
    row0 = DataParam.row0;
    col0 = DataParam.col0;
    theta0 = DataParam.theta0;
    DataParamDer = DataParam;
    SIZE = size(input_args,2);
    Grid = createGrid(rmin,rmax,theta0,row0,col0,SIZE);
    [output_args_dzdx,dzdGrid] = vl_nnbilinearsampler(input_args,Grid,dzdout);
    [~,DataParamDer.dzdrmin...
        ,DataParamDer.dzdrmax...
        ,DataParamDer.dzdtheta0...
        ,DataParamDer.dzdrow0...
        ,DataParamDer.dzdcol0] = createGrid(rmin,rmax,theta0,row0,col0,SIZE,dzdGrid);
end
end

function [grid,dzdrmin,dzdrmax,dzdtheta0,dzdrow0,dzdcol0] =  createGrid(rmin,rmax,theta0,row0,col0,SIZE,dzdGrid)
% rmin and rmax are of size 1*1*1*B
% SIZE is scalar
% dzdGrid 2*SIZE*SIZE*B
% gridR
doder = nargout>1;
if ~doder
    %% results
    LinspaceR = createLinR(rmin,rmax,SIZE);
    %LinspaceR is SIZE *1 *1*B
    LinspaceTheta = createLinTheta(theta0,SIZE);
    %LinspaceTheta is 1*SIZE *1*B
    rowCoordinates = bsxfun(@times,LinspaceR,cos(LinspaceTheta));
    % rowCoordinates is SIZE('R')*SIZE('Theta')*1*B
    colCoordinates = bsxfun(@times,LinspaceR,sin(LinspaceTheta));
    % colCoordinates is SIZE('R')*SIZE('Theta')*1*B
    rowCoordinates = bsxfun(@plus,rowCoordinates,row0);
    colCoordinates = bsxfun(@plus,colCoordinates,col0);
    % row and col coord is S*S*1*B
    sq_row_c = permute(rowCoordinates,[3,1,2,4]);
    sq_col_c = squeeze(colCoordinates,[3,1,2,4]);
    grid = cat(1,sq_row_c,sq_col_c);
else
    LinSpaceR = createLinR(rmin,rmax,SIZE);
    LinSpaceTheta = createLinTheta(theta0,SIZE);
    row_d = permute(dzdGrid(1,:,:,:),[2,3,1,4]);
    % sq_row_d  is S*S * 1 *B
    col_d = permute(dzdGrid(2,:,:,:),[2,3,1,4]);
    % sq_col_d  is S*S * 1 *B
    dzdrow0 = sum(sum(row_d,1),2);
    dzdcol0 = sum(sum(col_d,1),2);
    
    dzdrowCoordinates = row_d;
    dzdcolCoordinates = col_d;
    dzdLinSpaceR_1 = bsxfun(@times,dzdrowCoordinates,cos(LinSpaceTheta));
    dzdLinSpaceR_1 = sum(dzdLinSpaceR_1,2);
    
    dzdLinSpaceR_2 = bsxfun(@times,dzdcolCoordinates,sin(LinSpaceTheta));
    dzdLinSpaceR_2 = sum(dzdLinSpaceR_2,2);
    
    dzdLinSpaceTheta_1 = bsxfun(@times,LinSpaceR,-sin(LinSpaceTheta));
    dzdLinSpaceTheta_1 = sum(1,dzdLinSpaceTheta_1);
    
    dzdLinSpaceTheta_2 = bsxfun(@times,LinSpaceR,cos(LinSpaceTheta));
    dzdLinSpaceTheta_2 = sum(1,dzdLinSpaceTheta_2);
    
    dzdLinSpaceR = dzdLinSpaceR_1 + dzdLinSpaceR_2;
    dzdLinSpaceTheta = dzdLinSpaceTheta_1 + dzdLinSpaceTheta_2;
    
    [~,dzdtheta0] = createLinTheta(theta0,SIZE,dzdLinSpaceTheta);
    [~,dzdrmin,dzdrmax] = createLinR(rmin,rmax,SIZE,dzdLinSpaceR);
    
end
end
function [Linspace,dzdtheta0] = createLinTheta(theta0,SIZE,dzdLinSpace)
    % SIZE is scalar
    % theta0 is 1*1*1*B
    % LinSpace is 1*SIZE*1*B
    % dzdLinSpace is 1*SIZE*1*B
    doder = nargout>1;
    range = 2*pi;
    if ~doder
        %% calculate results
        % Linspace = theta0 + (i-1)*(2*pi)/(SIZE-1);
        iArray = gpuArray.colon(0,SIZE-1)*range/(SIZE-1);
        % iArray is 1*SIZE
        assert(ndims(theta0) == 4 | numel(theta0)==1, 'dimensions of rmin must be 4 instead : %d',ndims(theta0));
        Linspace = bsxfun(@pluse,iArray,theta0);
        
    else
        %% calculate derivative
        % dLinspace(i)dtheta0 = 1;
        dzdtheta0 = sum(2,dzdLinSpace);
    end
end
function [LinSpace,dzdrmin,dzdrmax] = createLinR(rmin,rmax,SIZE,dzdLinSpace)
    % rmin is 1*1*1*b
    % rmax is 1*1*1*b
    % dzdLinSpace is SIZE*1*1*B
    % LinSpace is SIZE*1*1*B
    % dzdrmin is 1*B
    % dzdrmax is 1*B
    doder = nargout>1;
    if doder
        %% calculate Derivative
        %dLinSpace(i)_drmin = 1 + (1-i)*/(SIZE-1);
        iArray = gpuArray.colon(0,SIZE-1)'/(SIZE-1);
        %iArray is SIZE * 1;
        iArray_max = bsxfun(@plus,iArray,1);
        iArray_min = bsxfun(@plus,-iArray,1);
        
        iArray_max_projected = bsxfun(@times,iArray_max, dzdLinSpace);
        iArray_min_projected = bsxfun(@times,iArray_min,dzdLinSpace);
        
        dzdrmin = sum(1,iArray_min_projected);
        dzdrmax = sum(1,iArray_max_projected);
    else
        %% calculate Results
        % LinSpace(i)= rmin + (i-1)*(rmax - rmin)/(SIZE-1);
        assert(ndims(rmin) == 4 | numel(rmin) == 1,...
            'dimensions of rmin must be 4 instead : %d',ndims(rmin));
        assert(ndims(rmax) == 4 | numel(rmin) == 1,...
            'dimensions of rmax must be 4 instead : %d',ndims(rmin));
        Step = rmax - rmin;
        %Step is 1*1*1*B
        LinSpaceBeans = gpuArray.colon(0,(SIZE-1))';
        %linspace beans is SIZE*1;
        LinSpace = bsxfun(@times,LinSpaceBeans,Step/(SIZE-1));
        %Linspace is SIZE*1*1*B
        LinSpace = bsxfun(@plus,LinSpace,rmin);
        %Linspace is SIZE*1*1*B
    end
    
end

