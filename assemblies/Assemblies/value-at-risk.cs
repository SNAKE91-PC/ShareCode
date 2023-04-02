using System;
using System.Threading;
using System.Linq;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;

namespace  Assemblies
{
    public class ValueAtRisk
    {
        //private readonly Instrument _instrument;
        private int window;
        private string methodology;

        public ValueAtRisk(int window, string methodology = "historical")
        {
            this.window = window;
            this.methodology = methodology;

        }

        public double getVaR(Instrument instrument)
        {
            riskFactor[] rfList = instrument.curves;
            DataTable dt            = new DataTable() ;
            int  rows_returned ;

            // const string credentials = @"Server=(localdb)\.\PDATA_SQLEXPRESS;Database=PDATA_SQLEXPRESS;User ID=sa;Password=2BeChanged!;" ;
            // const string sqlQuery = @"
            //       select tPatCulIntPatIDPk ,
            //              tPatSFirstname    ,
            //              tPatSName         ,
            //              tPatDBirthday
            //       from dbo.TPatientRaw
            //       where tPatSName = @patientSurname
            //       " ;
            //
            // using ( SqlConnection connection = new SqlConnection(credentials) )
            // using ( SqlCommand    cmd        = connection.CreateCommand() )
            // using ( SqlDataAdapter sda       = new SqlDataAdapter( cmd ) )
            // {
            //     cmd.CommandText = sqlQuery ;
            //     cmd.CommandType = CommandType.Text ;
            //     connection.Open() ;
            //     rows_returned = sda.Fill(dt) ;
            //     connection.Close() ;
            // }
            //     
            return 0;
        }
        
    }
    
}
