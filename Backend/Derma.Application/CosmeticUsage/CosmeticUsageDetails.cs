using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Derma.Domain;
using Derma.Infrastructure;
using Microsoft.EntityFrameworkCore;
namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageDetails
    {
        public class Query : IRequest<Derma.Domain.CosmeticUsage?>
        {
            public Guid Id { get; set; }
            public Guid UserId { get; set; }
            public bool IsAdmin { get; set; }
        }

        public class Handler : IRequestHandler<Query, Derma.Domain.CosmeticUsage>
        { 
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<Derma.Domain.CosmeticUsage?> Handle(Query request, CancellationToken cancellationToken)
            {
                var query = _context.CosmeticUsage.Where(c => c.Id == request.Id);

                if (!request.IsAdmin)
                    query = query.Where(c => c.userId == request.UserId);

                return await query.FirstOrDefaultAsync(cancellationToken);
            }
        }
    }
}
