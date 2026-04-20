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
    public class CosmeticUsageList
    {
        public class Query : IRequest<List<Derma.Domain.CosmeticUsage>> { }

        public class Handler : IRequestHandler<Query, List<Derma.Domain.CosmeticUsage>> {
            private readonly DataContext _context;
            public Handler(DataContext context) {
                _context = context;
            }

            public async Task<List<Derma.Domain.CosmeticUsage>> Handle(Query request, CancellationToken cancellationToken) {
                return await _context.CosmeticUsage.ToListAsync();
            }
        }
    }
}
