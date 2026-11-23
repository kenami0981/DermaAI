using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageEdit
    {
        public class Command : IRequest<bool>
        {
            public required Derma.Domain.CosmeticUsage CosmeticUsage { get; set; }
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }
        public class Handler : IRequestHandler<Command, bool> {
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<bool> Handle(Command request, CancellationToken cancellationToken)
            {
                var query = _context.CosmeticUsage.Where(c => c.Id == request.CosmeticUsage.Id);

                if (!request.IsAdmin)
                    query = query.Where(c => c.userId == request.UserId);

                var cosmeticUsage = await query.FirstOrDefaultAsync(cancellationToken);
                if (cosmeticUsage == null) return false;

                cosmeticUsage.productName = request.CosmeticUsage.productName;
                cosmeticUsage.category = request.CosmeticUsage.category;
                cosmeticUsage.startDate = request.CosmeticUsage.startDate;
                cosmeticUsage.endDate = request.CosmeticUsage.endDate; 
                cosmeticUsage.frequency = request.CosmeticUsage.frequency;
                cosmeticUsage.notes = request.CosmeticUsage.notes;
               

                await _context.SaveChangesAsync(cancellationToken);
                return true;
            }
        }   
    }
}
