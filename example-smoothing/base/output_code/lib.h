#pragma once
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <LBFGS.h>

struct MeshSmoothing {
    double ret;
    Eigen::MatrixXd V;
    Eigen::SparseMatrix<double> L;
    double λ;
    double E_textdata(
        const Eigen::MatrixXd & U)
    {
        const long n = U.rows();
        assert( U.cols() == 3 );

        return pow((U - V).norm(), 2);    
    }
    double E_textsmoothness(
        const Eigen::MatrixXd & U)
    {
        const long n = U.rows();
        assert( U.cols() == 3 );

        return (U.transpose() * L * U).trace();    
    }
    MeshSmoothing(
        const Eigen::MatrixXd & V,
        const Eigen::SparseMatrix<double> & L,
        const double & λ)
    {
        const long n = V.rows();
        assert( V.cols() == 3 );
        assert( L.rows() == n );
        assert( L.cols() == n );
        this->V = V;
        this->L = L;
        this->λ = λ;
        auto pack = [&](Eigen::MatrixXd & U)
        {
            Eigen::VectorXd vec_joined(3*n);
            vec_joined << Eigen::Map<Eigen::VectorXd>(U.data(), U.cols()*U.rows());
            return vec_joined;
        };
        auto unpack = [&](Eigen::VectorXd & X_1, Eigen::MatrixXd & U)
        {
            double* a = X_1(Eigen::seqN(0, 3*n)).data();
            U = Eigen::Map<Eigen::MatrixXd >(a);
        };
        LBFGSpp::LBFGSParam<double> param;
        param.epsilon = 1e-6;
        param.max_iterations = 100;
        LBFGSpp::LBFGSSolver<double> solver(param); 
        auto target_1 = [&](Eigen::VectorXd & X_1)
        {
            Eigen::MatrixXd U;
            unpack(X_1, U);
            return E_textdata(U) + λ * E_textsmoothness(U);
        };
        double eps = 1e-6;
        auto helper = [&](Eigen::VectorXd & X_1, Eigen::VectorXd & grad)
        {
            double f_X = target_1(X_1);
            for(int i = 0; i < grad.size(); ++i )
            {
                const double X_1_i = X_1[i];
                X_1[i] += eps;
                grad[i] = (target_1(X_1) - f_X)/eps;
                X_1[i] = X_1_i;
            }
            return f_X;
        };
        Eigen::VectorXd init = Eigen::VectorXd::Zero(3*n);
        double fx;
        ret = solver.minimize(helper, init, fx);
    }
};

