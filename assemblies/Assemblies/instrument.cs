using System;
using System.Collections.Generic;
using Microsoft.Data.Analysis;

namespace Assemblies
{
    public abstract class Instrument
    {
        public double pv;
        public riskFactor[] curves;
        public DateTime date;
    }

}