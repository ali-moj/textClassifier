package com.ali.textClassifier.utils;

import java.util.List;

public class Calculator {


    public static double entropy(List<Double> probabilities) {
        // H(X) = -∑P(x∈X)logP(x∈X), H(X|y) = -∑P(x∈X|y)logP(x∈X|y)
        return probabilities.stream().filter(p -> p > 0.0).mapToDouble(p -> -p * Math.log(p)).sum();
    }

    public static double conditionalEntrogy(double probability, List<Double> PconditionWithFeature,
                                            List<Double> PconditionWithoutFeature) {
        // H(X|Y) = P(y=1.txt)H(X|y) + P(y=0)H(X|y)
        return probability * entropy(PconditionWithFeature) + (1 - probability) * entropy(PconditionWithoutFeature);
    }


    public static double chisquare(int A, int B, int C, int D) {
        // chi = n*(ad-bc)^2/(a+c)*(b+d)*(a+b)*(c+d)
        double chi = Math.log(A + B + C + D) + 2 * Math.log(Math.abs(A * D - B * C))
                - (Math.log(A + C) + Math.log(B + D) + Math.log(A + B) + Math.log(C + D));
        return Math.exp(chi);
    }


    public static double Ppost(double Pprior, final List<Double> Pconditions) {
        return Pprior + Pconditions.stream().mapToDouble(Double::valueOf).sum();
    }
}
