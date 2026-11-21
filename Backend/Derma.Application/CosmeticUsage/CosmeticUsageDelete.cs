
using Derma.Infrastructure;
using MediatR;
using Microsoft.EntityFrameworkCore;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageDelete
    {
        public class Command : IRequest<bool>
        {
            public Guid Id { get; set; }
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }

        public class Handler : IRequestHandler<Command, bool>
        {
            private readonly DataContext _context;

            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<bool> Handle(Command request, CancellationToken cancellationToken)
            {
                var query = _context.CosmeticUsage.Where(c => c.Id == request.Id);

                if (!request.IsAdmin)
                    query = query.Where(c => c.userId == request.UserId);

                var cosmeticUsage = await query.FirstOrDefaultAsync(cancellationToken);
                if (cosmeticUsage == null) return false;

                _context.CosmeticUsage.Remove(cosmeticUsage);
                await _context.SaveChangesAsync(cancellationToken);
                return true;
            }
        }
    }
}
