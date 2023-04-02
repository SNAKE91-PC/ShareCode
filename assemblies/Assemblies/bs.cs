using System;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using System.Collections.Generic;

namespace Assemblies
{
    public class BlackScholes
    {

        public double simulatePrice(double r, double S0, double sigma, double T)
        {

            Normal normDistr = new Normal(0, 1);
            double St = S0 * Math.Exp(r - 0.5 * Math.Pow(sigma, 2) * T + sigma * normDistr.InverseCumulativeDistribution(new Random().NextDouble()) * Math.Pow(T, 2));

            return St;

        }

        public double[] simulatePrice(double[] r, double[] S0, double[] sigma, double[] T)
        {
            double[] St = new double[r.Length];
            Parallel.For(0, r.Length, i =>
            {
                St[i] = simulatePrice(r[i], S0[i], sigma[i], T[i]);
            });

            return St;

        }

    }


    public class OptionBuilderEquity
    {
        public OptionEquity getOption(string OptionType, Dictionary<string,double> paramsList)
        {
            
            switch (OptionType)
            {

                case "European": return new OptionEUEquity(paramsList);
                case "American": return new OptionUSEquity(paramsList);
                case "Asian": return new OptionASEquity(paramsList);

                default:
                    {
                        throw new Exception("Option Type not supported");
                    }

            }
        }
    }

    public abstract class OptionEquity
    {
        protected double r, S0, sigma, T, K;
        public OptionEquity(Dictionary<string, double> paramsList)
        {
            try
            {
                r = paramsList["r"];
                S0 = paramsList["S0"];
                sigma = paramsList["sigma"];
                T = paramsList["T"];
            }
            catch
            {
                throw new Exception("Please set all parameters (r, S0, sigma, T)");
            }
        }

        public virtual double payoff() { return -1; }
        public virtual double pv() { return -1; }
    }


    public class OptionEUEquity : OptionEquity
    {
        public OptionEUEquity(Dictionary<string, double> paramsList) : base(paramsList) { }

        public override double payoff()
        {

            return Math.Max(S0 - K, 0);
        }

        public override double pv()
        {
            return base.pv();
        }
    }

    public class OptionUSEquity : OptionEquity
    {
            public BlackScholes rng;
            public double[] pathSt;
        public OptionUSEquity(Dictionary<string, double> paramsList) : base(paramsList) { rng = new BlackScholes(); }

        public override double payoff()
        {

                return 1;
        }

        public override double pv()
        {
            return base.pv();
        }
    }


    public class OptionASEquity : OptionEquity
    {
            public BlackScholes rng;
            public double[] pathSt;
        public OptionASEquity(Dictionary<string, double> paramsList) : base(paramsList) { }
        public override double payoff()
        {

            return -1;
        }


        public override double pv()
        {
            return base.pv();
        }



    }




}
