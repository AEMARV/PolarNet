function y_dzdx = vl_nndealchannel(layer,resi,dzdy)
    in = resi.x;
    SIZE = size(in);
    batchsize = layer.batchsize;
    dealback = false;
    if SIZE(4) ~= batchsize
        dealback = true;
    end
    if nargin<3
        if dealback
        y_dzdx = reshape(resi.x,SIZE(1),SIZE(2),[],batchsize);
        else
            y_dzdx = reshape(resi.x,SIZE(1),SIZE(2),1,[]);
        end
    else
        if dealback
        y_dzdx = reshape(dzdy,SIZE(1),SIZE(2),1,[]);
        else
            y_dzdx = reshape(dzdy,SIZE(1),SIZE(2),[],batchsize);
        end
    end
    
end