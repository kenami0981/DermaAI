using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Derma.Domain;
using Derma.Infrastructure;
namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageDetails
    {
        public class Query : IRequest<Derma.Domain.CosmeticUsage> { 
        public Guid Id { get; set; }
        }

        public class Handler : IRequestHandler<Query, Derma.Domain.CosmeticUsage>
        { 
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<Derma.Domain.CosmeticUsage> Handle(Query request, CancellationToken cancellationToken) {
                return await _context.CosmeticUsage.FindAsync(request.Id);
            }
        }
    }
}
