using Derma.Infrastructure;
using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Derma.Domain;
using Microsoft.EntityFrameworkCore;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisList
    {
        public class Query : IRequest<List<Derma.Domain.SkinAnalysis>> { }

        public class Handler : IRequestHandler<Query, List<Derma.Domain.SkinAnalysis>>
        {
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }

            public async Task<List<Derma.Domain.SkinAnalysis>> Handle(Query request, CancellationToken cancellationToken)
            {
                return await _context.SkinAnalysis.ToListAsync();
            }
        }
    }
}
