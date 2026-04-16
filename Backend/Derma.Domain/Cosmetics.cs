using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Domain
{
    public enum Category
    {
        Cleanser,
        Toner,
        Moisturizer, 
        Serum,       
        Sunscreen,     
        Treatment,
        Other
    }
    public class CosmeticUsage
    {
        public Guid Id { get; set; }
        public Guid userId { get; set; }
        public required string productName { get; set; }
        public required Category category { get; set; }
        public DateTime startDate { get; set; }
        public DateTime? endDate { get; set; }
        public string? frequency { get; set; }
        public string? notes { get; set; }
    }
    

}
