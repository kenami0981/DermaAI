using Derma.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Infrastructure
{
    public class Seed
    {
        public static async Task SeedData(DataContext context) {
            if (context.CosmeticUsage.Any()) return;
            var userId = Guid.NewGuid();
            var cosmeticUsage = new List<CosmeticUsage> {
                new CosmeticUsage {
                   Id = Guid.NewGuid(),
                    userId = userId,
                    productName = "CeraVe Foaming Cleanser",
                    category = Category.Cleanser,
                    startDate = new DateTime(2025, 01, 10),
                    endDate = null,
                    frequency = "Twice daily",
                    notes = "No irritation, skin feels less oily"
                },
            };
            await context.CosmeticUsage.AddRangeAsync(cosmeticUsage);
            await context.SaveChangesAsync();
        }
    }
}
