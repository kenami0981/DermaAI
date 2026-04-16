using Derma.Domain;
using Derma.Infrastructure;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Derma.API.Controllers
{
    public class CosmeticUsageController : BaseApiController
    {
        private readonly DataContext _context;
        public CosmeticUsageController(DataContext context)
        {
            _context = context;
        }

        [HttpGet]
        public async Task<ActionResult<List<CosmeticUsage>>> GetCosmeticsUsage() {
            return await _context.CosmeticUsage.ToListAsync();
        }
        [HttpGet("{id}")]
        public async Task<ActionResult<CosmeticUsage>> GetCosmeticUsage(Guid id)
        {
            return await _context.CosmeticUsage.FindAsync(id);
        }
    }
}
