using Derma.Infrastructure;
using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageEdit
    {
        public class Command : IRequest<Unit>
        {
            public required Derma.Domain.CosmeticUsage CosmeticUsage { get; set; }
        }
        public class Handler : IRequestHandler<Command, Unit> {
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<Unit> Handle(Command request, CancellationToken cancellationToken) {
                var cosmeticUsage = await _context.CosmeticUsage.FindAsync(request.CosmeticUsage.Id);
                cosmeticUsage.productName = request.CosmeticUsage.productName;
                cosmeticUsage.frequency = request.CosmeticUsage.frequency ?? cosmeticUsage.frequency;
                cosmeticUsage.notes = request.CosmeticUsage.notes ?? cosmeticUsage.notes;
                cosmeticUsage.startDate = request.CosmeticUsage.startDate;
                cosmeticUsage.endDate = request.CosmeticUsage.endDate ?? cosmeticUsage.endDate;
                cosmeticUsage.category = request.CosmeticUsage.category;
                request.CosmeticUsage.userId = cosmeticUsage.userId;
                await _context.SaveChangesAsync();

                return Unit.Value;

            }
        }   
    }
}
