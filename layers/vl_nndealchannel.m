function y_dzdx = vl_nndealchannel(layer,resi,dzdy)
    in = resi.x;
    SIZE = in;
    if nargin<3
        y_dzdx = reshape(resi.x,SIZE(1),SIZE(2),1,[]);
    else
        y_dzdx = reshape(dzdy,SIZE(1),SIZE(2),SIZE(3),SIZE(4));
    end
    
end