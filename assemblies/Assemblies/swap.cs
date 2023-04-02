using System;

namespace Assemblies;
using System.Collections.Generic;

public abstract class ISwap
{
    public abstract double getPV();
    public abstract double getParRate();

}


public class VanillaSwap : ISwap
{
    private readonly double ParRate;
    private readonly string forwardCurve;
    private Dictionary<DateTime, double> map;

    
    
    VanillaSwap()
    {
        
    }
    
    public override double getPV()
    {
        return 0;
    }

    public override double getParRate()
    {
        return 0;
    }

    private Dictionary<string, double> getForwardCurve()
    {
        Dictionary<string, double> map = new Dictionary<DateTime, double>();

        return map;
    }
    
    
}