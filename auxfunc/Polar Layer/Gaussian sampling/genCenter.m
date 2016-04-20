function centers = genCenter(History)
% centers = genCenter(History)
% samples new centers based on the History of previous centers
% -------------------------------------------------------------------------
% INPUTS:
%
% History: is matrix of size 2*2*M*N where M represents number of previous
% centers
% N is the Batch size
% firest row is center coordinates (means of the gaussian) normalized
% between 0:1
% second row is sigmaX and sigmaY which is normalized between 0:1
%
%--------------------------------------------------------------------------
% OUTPUTS: 
% centers is a matrix of size N*2
% first column is the Xs, second column is the Ys
% normalized between 0:1

%% INITIAL PARAMS
if nargin >=2
X_size = ImageSize(1);
Y_size = ImageSize(2);
end
BATCH_SIZE = size(History,4);
NUM_CEN = size(History,3);
ONES_BATCH = ones(1,BATCH_SIZE);
NUM_SAMPLES = 1;
NumSamplesCell = mat2cell(ones(1,BATCH_SIZE),1,ONES_BATCH);
%PLOT_ON = 0;
%% Get sub matrices
VarxMat = History(2,1,:,:);
%VarxMat is 1*1*M*N;
VaryMat = History(2,2,:,:);
%VaryMat is 1*1*M*N;
XmeanMat = History(1,1,:,:);
YmeanMat = History(1,2,:,:);
% X and Y mean are 1*1*M*N
% remove singleton dimensions\
% divide Variance by 6 cuz they are normalized 0:1
VarxMat = squeeze(VarxMat)/6;
VaryMat = squeeze(VaryMat)/6;
XmeanMat = squeeze(XmeanMat);
YmeanMat = squeeze(YmeanMat);
%everything above has dimension of NUM_CEN * BATCH_SIZE
% Convert everything to cells of size 1*BATCH_SIZE
% each entry has a matrix of size NUM_CEN * 1 
varxCell = mat2cell(VarxMat,[NUM_CEN],ONES_BATCH);
varYCell = mat2cell(VaryMat,[NUM_CEN],ONES_BATCH);
meanXCell = mat2cell(XmeanMat,[NUM_CEN],ONES_BATCH);
meanYCell = mat2cell(YmeanMat,[NUM_CEN],ONES_BATCH);
% pass the inputs to sampling function
newCenterX = cellfun(@UMGRN,meanXCell,varxCell,NumSamplesCell,'UniformOutput',true);
newCenterY = cellfun(@UMGRN,meanYCell,varYCell,NumSamplesCell,'UniformOutput',true);
centers = [newCenterX;newCenterY]';

end


