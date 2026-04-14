using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Derma.Domain;

namespace Derma.Infrastructure
{
    public class DataContext : DbContext
    {
        public DataContext(DbContextOptions<DataContext> options) : base(options) { }
        public DbSet<CosmeticUsage> CosmeticUsage { get; set; }
        public DbSet<SkinAnalysis> SkinAnalysis { get; set; }
    }
}
