using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Derma.Domain;

namespace Derma.Infrastructure
{
    public class DataContext : IdentityDbContext<AppUser, IdentityRole<Guid>, Guid>
    {
        public DataContext(DbContextOptions<DataContext> options) : base(options) { }

        public DbSet<CosmeticUsage> CosmeticUsage { get; set; }
        public DbSet<SkinAnalysis> SkinAnalysis { get; set; }
        public DbSet<AnalysisResult> AnalysisResult { get; set; }
        public DbSet<SkinAnalysisTag> SkinAnalysisTag { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<SkinAnalysisTag>()
                .Property(x => x.Tag)
                .HasConversion<string>();

            modelBuilder.Entity<SkinAnalysisTag>()
                .HasIndex(x => new { x.SkinAnalysisId, x.Tag })
                .IsUnique();

            modelBuilder.Entity<SkinAnalysis>()
                .HasOne(a => a.Result)
                .WithOne()
                .HasForeignKey<AnalysisResult>(r => r.skinAnalysisId)
                .OnDelete(DeleteBehavior.Cascade);
        }
    }
}