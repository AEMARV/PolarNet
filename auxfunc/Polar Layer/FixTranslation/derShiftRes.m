function Resout = derShiftRes(res)
% function [drow,dcol] = derShiftRes(res,net)
% fix res calculates the derivative of the loss function with respect to
% the shifts in each layer of the results
    Resout = res;
   for i = numel(res)-1 : -1 : 1
        if isempty(res(i).dzdx)
            continue;
        end
        [dzdrow,dzdcol] = findDer(res(i));
        Resout(i).dzdrow = dzdrow;
        Resout(i).dzdcol = dzdcol;
        
   
   end

end
function [dzdrow,dzdcol] = findDer(resi)
x= resi.x;
dzdx = resi.dzdx;
if size(dzdx,1) <3 | size(dzdx,2)<3
%error('not Implemented');
end
[gr,gc] = BatchGradient(x,0,'circular');
[dzdrow,dzdcol] = findDir(gr,gc,dzdx);

end

function [dzdrow,dzdcol] = findDir(gr,gc,dzdx)
el_times = dzdx .* gr;
% el_times is the element wise product of dzdx and Polar_grad_row
dzdrow = sum(sum(sum(el_times,1),2),3);
% dz drow should be of size B*1
dzdrow = squeeze(dzdrow);
el_times = dzdx .* gc;
% el_times is the element wise product of dzdx and Polar_grad_row
dzdcol = sum(sum(sum(el_times,1),2),3);
dzdcol = squeeze(dzdcol);
end
