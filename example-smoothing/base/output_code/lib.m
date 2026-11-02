function output = MeshSmoothing(V, L, lambda_)
% output = MeshSmoothing(V, L, λ)
%
%    `E_\text{data}`(U) = ‖ U - V ‖² where U ∈ ℝ^(n×3)
%    `E_\text{smoothness}`(U) = trace( Uᵀ L U ) where U ∈ ℝ^(n×3)
%    trace from linearalgebra
%    
%    min_( U ∈ ℝ^(n×3) ) `E_\text{data}`(U) + λ `E_\text{smoothness}`(U)
%    where
%    V ∈ ℝ^(n×3): The vertices of the mesh
%    L ∈ ℝ^(n×n) sparse: The Laplacian matrix
%    λ ∈ ℝ
%    
    if nargin==0
        warning('generating random input data');
        [V, L, lambda_] = generateRandomData();
    end
    function [V, L, lambda_] = generateRandomData()
        lambda_ = randn();
        n = randi(10);
        V = randn(n, 3);
        L = sparse.random(n, n, dtype=np.float64, density=0.25)
    end

    n = size(V, 1);
    assert( isequal(size(V), [n, 3]) );
    assert( isequal(size(L), [n, n]) );
    assert(numel(lambda_) == 1);

    U = optimvar('U', n, 3);
    target_2 = @(U) E_textdata(reshape(U, [n, 3])) + lambda_ * E_textsmoothness(reshape(U, [n, 3]));
    [~, optimize] = fminunc(target_2,zeros(n*3,1));
    ret = optimize;
    function [ret_2] = E_textdata(U)
        n = size(U, 1);
        assert( isequal(size(U), [n, 3]) );

        ret_2 = norm(U - V, 'fro').^2;
    end

    function [ret_3] = E_textsmoothness(U)
        n = size(U, 1);
        assert( isequal(size(U), [n, 3]) );

        ret_3 = trace(U' * L * U);
    end

    output.ret = ret;
    output.E_textdata = @E_textdata;
    output.E_textsmoothness = @E_textsmoothness;
    output.V = V;    
    output.L = L;    
    output.lambda_ = lambda_;
end

